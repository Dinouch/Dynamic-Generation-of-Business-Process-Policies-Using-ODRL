"""
policy_projection_agent.py — Agent 4 : Policy Projection Agent

Responsabilité unique :
    Générer les Fragment Policies (FPa et FPd) en JSON-LD ODRL
    conformément à la section 4.2 du rapport technique.

Corrections v2 :
    - FPd XOR/AND/OR générés depuis les PATTERNS du graphe formel
      (les connexions relient gateway→activité, pas activité→activité directement)
    - Méthode export() pour sérialiser en JSON-LD propre (sans clés _xxx)
      et écrire un fichier .jsonld par policy
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .structural_analyzer import EnrichedGraph, ConnectionInfo
from .constraint_validator import ValidationReport
from .implicit_dependency_detector import CandidateDependency, ImplicitDepType



# ─────────────────────────────────────────────
#  Helpers — génération d'URIs
# ─────────────────────────────────────────────

BASE_URI = "http://example.com"

def uri_policy(uid: str) -> str:
    return f"{BASE_URI}/policy:{uid}"

def uri_rule(name: str) -> str:
    return f"{BASE_URI}/rules/{name.replace(' ','_').replace('-','_')}"

def uri_asset(name: str) -> str:
    return f"{BASE_URI}/asset/{name.replace(' ','_').replace('-','_')}"

def uri_collection(name: str) -> str:
    return f"{BASE_URI}/assets/{name.replace(' ','_').replace('-','_')}"

def uri_message(fi: str, fj: str) -> str:
    return f"{BASE_URI}/messages/msg_{fi}_{fj}"

def new_uid() -> str:
    return str(uuid.uuid4())[:8]


# ─────────────────────────────────────────────
#  Structures de sortie
# ─────────────────────────────────────────────

@dataclass
class FragmentPolicySet:
    """
    Ensemble complet des policies d'un fragment fi.
    FPij(aij) = < idfi, FPa(aij), [FPd(aij, aik)] >
    """
    fragment_id:  str
    fpa_policies: list[dict] = field(default_factory=list)
    fpd_policies: list[dict] = field(default_factory=list)

    def all_policies(self) -> list[dict]:
        return self.fpa_policies + self.fpd_policies

    def to_odrl(self) -> list[dict]:
        """Retourne les policies en JSON-LD ODRL pur — sans clés internes _xxx."""
        return [{k: v for k, v in p.items() if not k.startswith("_")}
                for p in self.all_policies()]

    def summary(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "fpa_count":   len(self.fpa_policies),
            "fpd_count":   len(self.fpd_policies),
            "total":       len(self.all_policies()),
        }


# ─────────────────────────────────────────────
#  Agent 4 — Policy Projection Agent
# ─────────────────────────────────────────────

class PolicyProjectionAgent:

    def __init__(
        self,
        enriched_graph: EnrichedGraph,
        validation_report: Optional[ValidationReport] = None,
    ):
        self.enriched_graph    = enriched_graph
        self.validation_report = validation_report
        self._activity_rule_index:   dict[str, str] = {}
        self._activity_policy_index: dict[str, str] = {}

    # ─────────────────────────────────────────
    #  Point d'entrée — génération (séquentielle)
    # ─────────────────────────────────────────

    def generate(self) -> dict[str, FragmentPolicySet]:
        """Génération séquentielle (comportement historique)."""
        print("[Agent 4] Policy Projection Agent — démarrage de la génération")

        # Pré-indexer TOUTES les activités avant de générer les FPd inter-fragments
        self._preindex_all_activities()

        results: dict[str, FragmentPolicySet] = {}
        for fragment_id in self.enriched_graph.fragment_contexts.keys():
            fps = self._generate_fragment(fragment_id)
            results[fragment_id] = fps
        return results

    # ─────────────────────────────────────────
    #  Point d'entrée — génération (parallèle)
    # ─────────────────────────────────────────

    def generate_parallel(self, max_workers: Optional[int] = None) -> dict[str, FragmentPolicySet]:
        """
        Génère les policies "par fragment" en parallèle (ThreadPool).

        Notes :
        - On pré-indexe d'abord toutes les activités (global) pour que les FPd
          inter-fragments (message) puissent résoudre ruleix/rulejy.
        - Ensuite chaque fragment est généré indépendamment.
        """
        print("[Agent 4] Policy Projection Agent — génération parallèle")
        self._preindex_all_activities()

        frag_ids = list(self.enriched_graph.fragment_contexts.keys())
        results: dict[str, FragmentPolicySet] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._generate_fragment, frag_id): frag_id for frag_id in frag_ids}
            for fut in as_completed(futures):
                frag_id = futures[fut]
                fps = fut.result()
                results[frag_id] = fps

        # Conserver un ordre stable (même si dict est ordonné en Py3.7+, ici
        # on renvoie un dict reconstruit dans l'ordre des fragments initiaux)
        return {fid: results[fid] for fid in frag_ids if fid in results}

    # ─────────────────────────────────────────
    #  Génération par fragment (thread-safe)
    # ─────────────────────────────────────────

    def _generate_fragment(self, fragment_id: str) -> FragmentPolicySet:
        context = self.enriched_graph.fragment_contexts[fragment_id]
        print(f"[Agent 4] Génération des policies pour '{fragment_id}'...")

        fps = FragmentPolicySet(fragment_id=fragment_id)

        # Étape 1 — FPa pour chaque activité (pas d'écriture dans index global)
        for activity_name in context.activities:
            fpa = self._generate_fpa(activity_name, fragment_id)
            if fpa:
                fps.fpa_policies.append(fpa)

        # Étape 2a — FPd gateway XOR/AND/OR depuis les PATTERNS du graphe
        for pattern in self.enriched_graph.patterns:
            if pattern.fragment_id != fragment_id:
                continue
            if pattern.pattern_type not in ("fork_xor", "fork_and", "fork_or"):
                continue
            fps.fpd_policies.extend(self._fpd_from_pattern(pattern, fragment_id))

        # Étape 2b — FPd séquences internes
        for conn in context.connections:
            if conn.connection_type.lower() == "sequence" and not conn.is_inter:
                fpd = self._fpd_flow_sequence(conn, fragment_id)
                if fpd:
                    fps.fpd_policies.append(fpd)

        # Étape 2c — FPd message inter-fragment (downstream)
        for conn in context.downstream_deps:
            fpd = self._generate_fpd_message(conn, fragment_id)
            if fpd:
                fps.fpd_policies.append(fpd)

        # Étape 3 — FPd depuis dépendances implicites validées (Agent 3)
        if self.validation_report:
            for vr in self.validation_report.accepted:
                c = vr.candidate
                if c.source_fragment == fragment_id:
                    fpd = self._generate_fpd_from_implicit(c)
                    if fpd:
                        fps.fpd_policies.append(fpd)

        s = fps.summary()
        print(f"[Agent 4] '{fragment_id}' : "
              f"{s['fpa_count']} FPa + {s['fpd_count']} FPd = {s['total']} policies")

        return fps


    # ─────────────────────────────────────────
    #  Export JSON-LD ODRL
    # ─────────────────────────────────────────

    def export(
        self,
        fp_results: dict[str, FragmentPolicySet],
        output_dir: str = "./odrl_policies",
    ) -> dict[str, list[str]]:
        """
        Sérialise chaque policy en JSON-LD ODRL valide.
        Écrit un fichier .jsonld par policy dans output_dir/fragment_id/.
        Retourne dict[fragment_id → liste des chemins fichiers].
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: dict[str, list[str]] = {}

        for fragment_id, fps in fp_results.items():
            frag_dir = os.path.join(output_dir, fragment_id)
            os.makedirs(frag_dir, exist_ok=True)
            exported[fragment_id] = []

            for i, policy in enumerate(fps.all_policies()):
                # Nettoyer les clés internes _xxx → ODRL pur
                odrl = {k: v for k, v in policy.items() if not k.startswith("_")}

                ptype    = policy.get("_type", "FP")
                subtype  = (policy.get("_gateway") or policy.get("_flow") or
                            policy.get("_dep_type") or str(i))
                activity = (policy.get("_activity") or
                            "_".join(policy.get("_activities", [])[:1]))
                filename = f"{ptype}_{subtype}_{activity.replace('-','_')}.jsonld"
                filepath = os.path.join(frag_dir, filename)

                # Gérer les doublons de nom
                counter = 1
                base_path = filepath
                while os.path.exists(filepath):
                    filepath = base_path.replace(".jsonld", f"_{counter}.jsonld")
                    counter += 1

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(odrl, f, indent=2, ensure_ascii=False)
                exported[fragment_id].append(filepath)

        total = sum(len(v) for v in exported.values())
        print(f"[Agent 4] Export terminé : {total} fichiers .jsonld → '{output_dir}'")
        return exported

    # ─────────────────────────────────────────
    #  Pré-indexation globale
    # ─────────────────────────────────────────

    def _preindex_all_activities(self) -> None:
        """
        Pré-indexe rule_id et policy_uid pour TOUTES les activités.
        Nécessaire pour les FPd inter-fragments où l'activité cible
        est dans un autre fragment que la source.
        """
        for mapping in self.enriched_graph.b2p_mappings.values():
            act = mapping.activity_name
            if act in self._activity_rule_index:
                continue  # déjà indexé

            b2p = None
            if mapping.b2p_policy_ids:
                b2p = next(
                    (p for p in self.enriched_graph.raw_b2p
                     if p.get("uid") in mapping.b2p_policy_ids), None
                )

            if b2p:
                for rule_type in ("permission", "prohibition", "obligation"):
                    rules = b2p.get(rule_type, [])
                    if rules and rules[0].get("uid"):
                        self._activity_rule_index[act]   = rules[0]["uid"]
                        self._activity_policy_index[act] = b2p.get("uid", "")
                        break
            else:
                self._activity_rule_index[act]   = uri_rule(f"rule_{act.replace('-','_')}")
                self._activity_policy_index[act] = uri_policy(f"FPa_{act.replace('-','_')}")

    # ─────────────────────────────────────────
    #  FPa — Listing 4
    # ─────────────────────────────────────────

    def _generate_fpa(self, activity_name: str, fragment_id: str) -> Optional[dict]:
        mapping = next(
            (m for m in self.enriched_graph.b2p_mappings.values()
             if m.activity_name == activity_name), None
        )
        policy_uid = uri_policy(f"FPa_{activity_name.replace('-','_')}_{new_uid()}")

        # Cas 1 : B2P existante → projection directe
        if mapping and mapping.b2p_policy_ids:
            b2p = next(
                (p for p in self.enriched_graph.raw_b2p
                 if p.get("uid") in mapping.b2p_policy_ids), None
            )
            if b2p:
                fpa = {
                    "@context":     "http://www.w3.org/ns/odrl.jsonld",
                    "uid":          policy_uid,
                    "@type":        b2p.get("@type", "Set"),
                    "_fragment_id": fragment_id,
                    "_activity":    activity_name,
                    "_type":        "FPa",
                    "_source_b2p":  b2p.get("uid"),
                }
                for rule_type in ("permission", "prohibition", "obligation"):
                    if rule_type in b2p:
                        fpa[rule_type] = b2p[rule_type]
                return fpa

        # Cas 2 : policy minimale
        rule_uid = uri_rule(f"rule_{activity_name.replace('-','_')}")
        return {
            "@context":     "http://www.w3.org/ns/odrl.jsonld",
            "uid":          policy_uid,
            "@type":        "Set",
            "_fragment_id": fragment_id,
            "_activity":    activity_name,
            "_type":        "FPa",
            "_source_b2p":  None,
            "permission": [{
                "uid":    rule_uid,
                "target": uri_asset(activity_name),
                "action": "trigger",
                "constraint": [{
                    "leftOperand":  "dateTime",
                    "operator":     "gteq",
                    "rightOperand": {"@value": "2024-01-01", "@type": "xsd:date"}
                }]
            }]
        }

    def _extract_main_rule_id(self, fpa: dict, activity_name: str) -> str:
        for rule_type in ("permission", "prohibition", "obligation"):
            rules = fpa.get(rule_type, [])
            if rules and isinstance(rules, list) and rules[0].get("uid"):
                return rules[0]["uid"]
        return uri_rule(f"rule_{activity_name.replace('-','_')}")

    # ─────────────────────────────────────────
    #  FIX XOR : FPd depuis les patterns du graphe
    #  (et non depuis les connexions)
    # ─────────────────────────────────────────

    def _fpd_from_pattern(self, pattern, fragment_id: str) -> list[dict]:
        """
        Génère les FPd XOR/AND/OR depuis les patterns détectés par l'Agent 1.

        Pourquoi les patterns et pas les connexions ?
        Les connexions portent des arêtes gateway→activité.
        Pour générer enable(ruleij ⊕ ruleik), il faut les DEUX activités cibles
        d'un même gateway — ce que les patterns exposent directement via
        graph.out_edges(gateway_id).
        """
        graph   = self.enriched_graph.graph
        gw_node = graph.get_node(pattern.gateway_id)
        if not gw_node:
            return []

        out_edges = graph.out_edges(pattern.gateway_id)
        if len(out_edges) < 2:
            return []

        # Résoudre les branches : (activity_name, condition)
        branches = []
        for edge in out_edges:
            target = graph.get_node(edge.target)
            if target:
                branches.append({
                    "activity":  target.name,
                    "condition": edge.condition or f"condition_{target.name}",
                })

        gw_type  = pattern.pattern_type  # fork_xor / fork_and / fork_or
        policies = []

        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                b_i, b_j = branches[i], branches[j]
                if gw_type == "fork_xor":
                    policies.append(self._fpd_xor_pair(b_i, b_j, gw_node.name, fragment_id))
                elif gw_type == "fork_and":
                    policies.append(self._fpd_and_pair(b_i, b_j, gw_node.name, fragment_id))
                elif gw_type == "fork_or":
                    policies.append(self._fpd_or_pair(b_i, b_j, gw_node.name, fragment_id))

        return policies

    # ─────────────────────────────────────────
    #  gateway:XOR → Listing 7/8
    #  enable(ruleij ⊕ ruleik)
    # ─────────────────────────────────────────

    def _fpd_xor_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           uri_policy(f"FPd_XOR_{new_uid()}"),
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "XOR",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            "_conditions":   [b_i["condition"], b_j["condition"]],
            # Listing 7/8 — 2 permissions avec refinement conditionnel
            "permission": [
                {
                    "uid":    uri_rule(f"XOR_ij_{act_i}_{new_uid()}"),
                    "target": ruleij,
                    "action": [{
                        "rdf:value":  {"@id": "odrl:enable"},
                        "refinement": [{"leftOperand": "product",
                                        "operator":    "eq",
                                        "rightOperand": b_i["condition"]}]
                    }]
                },
                {
                    "uid":    uri_rule(f"XOR_ik_{act_j}_{new_uid()}"),
                    "target": ruleik,
                    "action": [{
                        "rdf:value":  {"@id": "odrl:enable"},
                        "refinement": [{"leftOperand": "product",
                                        "operator":    "eq",
                                        "rightOperand": b_j["condition"]}]
                    }]
                }
            ]
        }

    # ─────────────────────────────────────────
    #  gateway:AND → Listing 5
    #  enable(ruleij ∧ ruleik)
    # ─────────────────────────────────────────

    def _fpd_and_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))
        coll   = uri_collection(f"{act_i}_{act_j}_AND")

        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           uri_policy(f"FPd_AND_{new_uid()}"),
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "AND",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            # Listing 5
            "obligation": [{
                "uid":    uri_rule(f"AND_{act_i}_{act_j}_{new_uid()}"),
                "target": {"@type": "AssetCollection", "uid": coll},
                "action": "enable"
            }],
            "_asset_collection": [
                {"@type": "dc:Document", "@id": ruleij,
                 "dc:title": "concurrent rules", "odrl:partOf": coll},
                {"@type": "dc:Document", "@id": ruleik,
                 "dc:title": "concurrent rules", "odrl:partOf": coll},
            ]
        }

    # ─────────────────────────────────────────
    #  gateway:OR → Listing 6
    #  enable(ruleij ∨ ruleik)
    # ─────────────────────────────────────────

    def _fpd_or_pair(self, b_i: dict, b_j: dict, gw_name: str, fragment_id: str) -> dict:
        act_i  = b_i["activity"]
        act_j  = b_j["activity"]
        ruleij = self._activity_rule_index.get(act_i, uri_rule(f"rule_{act_i.replace('-','_')}"))
        ruleik = self._activity_rule_index.get(act_j, uri_rule(f"rule_{act_j.replace('-','_')}"))

        return {
            "@context":      "http://www.w3.org/ns/odrl.jsonld",
            "uid":           uri_policy(f"FPd_OR_{new_uid()}"),
            "@type":         "Set",
            "_fragment_id":  fragment_id,
            "_type":         "FPd",
            "_gateway":      "OR",
            "_gateway_name": gw_name,
            "_activities":   [act_i, act_j],
            # Listing 6
            "obligation": [
                {
                    "uid":    uri_rule(f"OR_ij_{act_i}_{new_uid()}"),
                    "target": ruleij,
                    "action": "enable",
                    "consequence": [{"target": ruleik, "action": "enable"}]
                },
                {
                    "uid":    uri_rule(f"OR_ik_{act_j}_{new_uid()}"),
                    "target": ruleik,
                    "action": "enable",
                    "consequence": [{"target": ruleij, "action": "enable"}]
                }
            ]
        }

    # ─────────────────────────────────────────
    #  flow:sequence → Listing 9
    #  enable(ruleij ≺ ruleik)
    # ─────────────────────────────────────────

    def _fpd_flow_sequence(self, conn: ConnectionInfo, fragment_id: str) -> dict:
        ruleij    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        policy_ik = self._activity_policy_index.get(
            conn.to_activity, uri_policy(f"FPa_{conn.to_activity.replace('-','_')}"))

        return {
            "@context":     "http://www.w3.org/ns/odrl.jsonld",
            "uid":          uri_policy(f"FPd_SEQ_{new_uid()}"),
            "@type":        "Set",
            "_fragment_id": fragment_id,
            "_type":        "FPd",
            "_flow":        "sequence",
            "_activities":  [conn.from_activity, conn.to_activity],
            # Listing 9
            "permission": [{
                "uid":    uri_rule(f"SEQ_{conn.from_activity}_{conn.to_activity}_{new_uid()}"),
                "target": ruleij,
                "action": "enable",
                "duty":   [{"action": "nextPolicy", "uid": policy_ik}],
                "constraint": [{
                    "leftOperand":  "event",
                    "operator":     "gt",
                    "rightOperand": {"@id": "odrl:policyUsage"}
                }]
            }]
        }

    # ─────────────────────────────────────────
    #  flow:message → Listing 10
    #  ruleix →msg→ enable(rulejy)
    # ─────────────────────────────────────────

    def _generate_fpd_message(self, conn, fragment_id: str) -> dict:
        ruleix    = self._activity_rule_index.get(
            conn.from_activity, uri_rule(f"rule_{conn.from_activity.replace('-','_')}"))
        rulejy    = self._activity_rule_index.get(
            conn.to_activity, uri_rule(f"rule_{conn.to_activity.replace('-','_')}"))
        from_frag = getattr(conn, "from_fragment", fragment_id)
        to_frag   = getattr(conn, "to_fragment",   "unknown")

        return {
            "@context":       "http://www.w3.org/ns/odrl.jsonld",
            "uid":            uri_policy(f"FPd_MSG_{new_uid()}"),
            "@type":          "Set",
            "_fragment_id":   fragment_id,
            "_type":          "FPd",
            "_flow":          "message",
            "_from_fragment": from_frag,
            "_to_fragment":   to_frag,
            "_activities":    [conn.from_activity, conn.to_activity],
            # Listing 10
            "permission": [{
                "uid":      uri_rule(f"MSG_{conn.from_activity}_{conn.to_activity}_{new_uid()}"),
                "target":   uri_message(from_frag, to_frag),
                "assignee": ruleix,
                "action": [{
                    "rdf:value":  {"@id": "odrl:transfer"},
                    "refinement": [{
                        "leftOperand":  "recipient",
                        "operator":     "eq",
                        "rightOperand": rulejy
                    }]
                }],
                "duty": [{
                    "target": rulejy,
                    "action": "enable",
                    "constraint": [{
                        "leftOperand":  "event",
                        "operator":     "gt",
                        "rightOperand": {"@id": "odrl:policyUsage"}
                    }]
                }]
            }]
        }

    # ─────────────────────────────────────────
    #  FPd implicite — depuis Agent 3
    # ─────────────────────────────────────────

    def _generate_fpd_from_implicit(self, candidate: CandidateDependency) -> dict:
        src_rule  = self._activity_rule_index.get(
            candidate.source_activity,
            uri_rule(f"rule_{candidate.source_activity.replace('-','_')}"))
        tgt_rule  = self._activity_rule_index.get(
            candidate.target_activity,
            uri_rule(f"rule_{candidate.target_activity.replace('-','_')}"))

        rule_type  = candidate.suggested_odrl_rule or "obligation"
        constraint = self._build_implicit_constraint(candidate)

        rule_body = {
            "uid":    uri_rule(f"IMP_{candidate.dep_type.value}_{new_uid()}"),
            "target": tgt_rule,
            "action": "enable",
        }
        if constraint:
            rule_body["constraint"] = [constraint]

        if rule_type == "prohibition":
            rule_body["target"]     = src_rule
            rule_body["constraint"] = [{
                "leftOperand":  "event",
                "operator":     "eq",
                "rightOperand": f"{candidate.target_activity}:active"
            }]

        return {
            "@context":       "http://www.w3.org/ns/odrl.jsonld",
            "uid":            uri_policy(f"FPd_IMP_{candidate.dep_type.value}_{new_uid()}"),
            "@type":          "Set",
            "_fragment_id":   candidate.source_fragment,
            "_type":          "FPd",
            "_implicit":      True,
            "_dep_type":      candidate.dep_type.value,
            "_confidence":    candidate.confidence,
            "_justification": candidate.justification,
            "_inter":         candidate.is_inter,
            rule_type:        [rule_body],
        }

    def _build_implicit_constraint(self, candidate: CandidateDependency) -> Optional[dict]:
        dep = candidate.dep_type
        if dep == ImplicitDepType.TEMPORAL:
            return {"leftOperand": "event", "operator": "gt",
                    "rightOperand": {"@id": "odrl:policyUsage"}}
        if dep == ImplicitDepType.ROLE:
            return {"leftOperand": "spatial", "operator": "eq",
                    "rightOperand": f"role:{candidate.source_activity}"}
        if dep == ImplicitDepType.COMPLIANCE:
            return {"leftOperand": "event", "operator": "eq",
                    "rightOperand": f"compliance:{candidate.source_activity}:completed"}
        if dep == ImplicitDepType.DATA:
            return {"leftOperand": "event", "operator": "eq",
                    "rightOperand": f"data:{candidate.source_activity}:produced"}
        if dep == ImplicitDepType.CONFLICT:
            return {"leftOperand": "event", "operator": "neq",
                    "rightOperand": f"{candidate.source_activity}:active"}
        return None