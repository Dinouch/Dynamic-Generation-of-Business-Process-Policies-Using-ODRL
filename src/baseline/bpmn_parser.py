"""
bpmn_parser.py

Parse BPMN XML vers le format interne aligné sur ``dataset/scenario1/bp_model.json`` :
activités (slugs, role Camunda, start/end fusionnés), gateways (slug, type), flux sequence
( from, to, type, gateway, condition optionnelle ), ``source_file``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any, Optional

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
CAMUNDA_NS = "http://camunda.org/schema/1.0/bpmn"


def slugify_label(name: str) -> str:
    """
    Identifiant kebab-case comme ``bp_model.json`` / ``fragments.json``
    (minuscules, tirets, ponctuation retirée).
    """
    s = (name or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def _slugify_condition(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    # Garder les opérateurs de comparaison sans les espacer avec des tirets
    s = re.sub(r"\s*([<>=!]+)\s*", r"\1", s)  # "> 500" → ">500"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w<>=!-]", "-", s, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or None


class BPMNParser:
    """
    BPMN → JSON aligné sur ``dataset/scenario1/bp_model.json``.
    """

    def __init__(self) -> None:
        self.namespaces = {
            "bpmn": BPMN_NS,
            "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
            "dc": "http://www.omg.org/spec/DD/20100524/DC",
            "di": "http://www.omg.org/spec/DD/20100524/DI",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "camunda": CAMUNDA_NS,
        }

    def parse_file(self, file_path: str) -> Optional[dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            model: dict[str, Any] = {
                "activities": [],
                "gateways": [],
                "flows": [],
                "source_file": os.path.basename(file_path),
                "_intermediate_events": [],
            }

            processes = root.findall(".//{%s}process" % BPMN_NS)
            if not processes:
                logger.warning("No process found in %s", file_path)
                return None

            for process in processes:
                self._extract_intermediate_events(process, model)
                self._extract_activities(process, model)
                self._extract_gateways(process, model)
                self._extract_flows(process, model)

            self._convert_ids_to_names(model)
            self._sync_intermediate_event_names(model)
            self._merge_start_end_events(model)
            self._collapse_intermediate_nodes(model)
            self._apply_slugify(model)
            self._finalize_dataset_shape(model)
            return model
        except Exception as e:
            logger.error("Error parsing %s: %s", file_path, e)
            return None

    def _camunda_assignee(self, el: ET.Element) -> Optional[str]:
        key = "{%s}assignee" % CAMUNDA_NS
        v = el.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
        return None

    def _extract_intermediate_events(self, process: ET.Element, model: dict[str, Any]) -> None:
        for tag in ("intermediateCatchEvent", "intermediateThrowEvent"):
            for el in process.findall(".//{%s}%s" % (BPMN_NS, tag)):
                eid = el.get("id")
                if not eid:
                    continue
                model["_intermediate_events"].append(
                    {
                        "id": eid,
                        "name": el.get("name") or eid,
                    }
                )

    def _extract_activities(self, process: ET.Element, model: dict[str, Any]) -> None:
        task_tags = (
            "task",
            "userTask",
            "serviceTask",
            "scriptTask",
            "manualTask",
            "businessRuleTask",
            "sendTask",
            "receiveTask",
        )
        seen_ids: set[str] = set()
        tasks: list[ET.Element] = []
        for tag in task_tags:
            for el in process.findall(".//{%s}%s" % (BPMN_NS, tag)):
                eid = el.get("id")
                if eid and eid in seen_ids:
                    continue
                if eid:
                    seen_ids.add(eid)
                tasks.append(el)

        start_events = process.findall(".//{%s}startEvent" % BPMN_NS)
        for event in start_events:
            model["activities"].append(
                {
                    "id": event.get("id"),
                    "name": event.get("name") or f"Start_{event.get('id')}",
                    "type": "event",
                    "start": True,
                }
            )

        end_events = process.findall(".//{%s}endEvent" % BPMN_NS)
        for event in end_events:
            model["activities"].append(
                {
                    "id": event.get("id"),
                    "name": event.get("name") or f"End_{event.get('id')}",
                    "type": "event",
                    "end": True,
                }
            )

        for task in tasks:
            act: dict[str, Any] = {
                "id": task.get("id"),
                "name": task.get("name") or f"Task_{task.get('id')}",
                "type": "task",
            }
            assignee = self._camunda_assignee(task)
            if assignee:
                act["role"] = assignee
            model["activities"].append(act)

    def _extract_gateways(self, process: ET.Element, model: dict[str, Any]) -> None:
        gateway_types = [
            ("exclusiveGateway", "XOR"),
            ("parallelGateway", "AND"),
            ("inclusiveGateway", "OR"),
            ("complexGateway", "COMPLEX"),
            ("eventBasedGateway", "EVENT_BASED_EXCLUSIVE"),
        ]
        for xml_type, internal_type in gateway_types:
            for gateway in process.findall(".//{%s}%s" % (BPMN_NS, xml_type)):
                model["gateways"].append(
                    {
                        "id": gateway.get("id"),
                        "name": gateway.get("name") or f"{internal_type}_{gateway.get('id')}",
                        "type": internal_type,
                    }
                )

    def _extract_flows(self, process: ET.Element, model: dict[str, Any]) -> None:
        flows = process.findall(".//{%s}sequenceFlow" % BPMN_NS)
        gateway_ids = {gw["id"]: gw for gw in model["gateways"]}

        for flow in flows:
            source_ref = flow.get("sourceRef")
            target_ref = flow.get("targetRef")
            gateway: Optional[str] = None
            if source_ref in gateway_ids:
                gateway = gateway_ids[source_ref]["name"]
            elif target_ref in gateway_ids:
                gateway = gateway_ids[target_ref]["name"]

            flow_obj: dict[str, Any] = {
                "id": flow.get("id"),
                "from": source_ref,
                "to": target_ref,
                "type": "sequence",
            }
            if gateway:
                flow_obj["gateway"] = gateway

            fname = flow.get("name")
            if fname and str(fname).strip():
                flow_obj["_condition_raw"] = str(fname).strip()

            ce = flow.find("{%s}conditionExpression" % BPMN_NS)
            if ce is not None and ce.text and ce.text.strip():
                flow_obj["_condition_raw"] = ce.text.strip()

            model["flows"].append(flow_obj)

        for flow in process.findall(".//{%s}messageFlow" % BPMN_NS):
            model["flows"].append(
                {
                    "id": flow.get("id"),
                    "from": flow.get("sourceRef"),
                    "to": flow.get("targetRef"),
                    "type": "message",
                }
            )

    def _build_id_lookup(self, model: dict[str, Any]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for act in model["activities"]:
            if act.get("id"):
                lookup[act["id"]] = act["name"]
        for gw in model["gateways"]:
            if gw.get("id"):
                lookup[gw["id"]] = gw["name"]
        for ie in model.get("_intermediate_events", []):
            if ie.get("id"):
                lookup[ie["id"]] = ie["name"]
        return lookup

    def _convert_ids_to_names(self, model: dict[str, Any]) -> None:
        lookup = self._build_id_lookup(model)
        for flow in model["flows"]:
            u, v = flow.get("from"), flow.get("to")
            if u in lookup:
                flow["from"] = lookup[u]
            if v in lookup:
                flow["to"] = lookup[v]
            gw = flow.get("gateway")
            if gw and gw in lookup:
                flow["gateway"] = lookup[gw]

    def _sync_intermediate_event_names(self, model: dict[str, Any]) -> None:
        """
        Aligne ``_intermediate_events[].name`` sur le même libellé que le lookup des flux
        (``_build_id_lookup`` / ``_convert_ids_to_names``), pour rester cohérent si l’ordre
        des passes change.
        """
        lookup = self._build_id_lookup(model)
        for ie in model.get("_intermediate_events", []):
            iid = ie.get("id")
            if iid and iid in lookup:
                ie["name"] = lookup[iid]

    @staticmethod
    def _seq_successors(seq_flows: list[dict[str, Any]], n: str) -> list[str]:
        return [f["to"] for f in seq_flows if f.get("from") == n]

    @staticmethod
    def _seq_predecessors(seq_flows: list[dict[str, Any]], n: str) -> list[str]:
        return [f["from"] for f in seq_flows if f.get("to") == n]

    def _merge_start_end_events(self, model: dict[str, Any]) -> None:
        """
        Propage ``start`` / ``end`` sur les tâches : au-delà d’un lien direct événement↔tâche,
        parcourt les gateways (et tout nœud non-tâche) jusqu’aux tâches, puis retire les
        événements start/end et les arcs incidents.
        """
        activities = model["activities"]
        by_name = {a["name"]: a for a in activities}
        seq_flows = [f for f in model["flows"] if f.get("type") == "sequence"]

        def is_task(n: str) -> bool:
            act = by_name.get(n)
            return bool(act and act.get("type") == "task")

        to_remove: set[str] = set()

        for a in list(activities):
            if a.get("type") != "event" or not a.get("start"):
                continue
            matched = False
            q: deque[str] = deque(self._seq_successors(seq_flows, a["name"]))
            seen: set[str] = set()
            while q:
                n = q.popleft()
                if n in seen:
                    continue
                seen.add(n)
                if is_task(n):
                    by_name[n]["start"] = True
                    matched = True
                    continue
                for nxt in self._seq_successors(seq_flows, n):
                    q.append(nxt)
            if matched:
                to_remove.add(a["name"])

        for a in list(activities):
            if a.get("type") != "event" or not a.get("end"):
                continue
            matched = False
            q = deque(self._seq_predecessors(seq_flows, a["name"]))
            seen = set()
            while q:
                n = q.popleft()
                if n in seen:
                    continue
                seen.add(n)
                if is_task(n):
                    by_name[n]["end"] = True
                    matched = True
                    continue
                for prv in self._seq_predecessors(seq_flows, n):
                    q.append(prv)
            if matched:
                to_remove.add(a["name"])

        model["activities"] = [a for a in activities if a["name"] not in to_remove]
        dead = to_remove
        model["flows"] = [
            f
            for f in model["flows"]
            if f.get("type") == "sequence"
            and f.get("from") not in dead
            and f.get("to") not in dead
        ] + [f for f in model["flows"] if f.get("type") != "sequence"]

    @staticmethod
    def _merge_flow_attrs(
        fi: dict[str, Any], fo: dict[str, Any], prefer_outgoing: bool = True
    ) -> dict[str, Any]:
        """Fusionne métadonnées gateway / condition pour un arc remplaçant (fi; n; fo)."""
        extra: dict[str, Any] = {}
        if prefer_outgoing:
            if fo.get("gateway"):
                extra["gateway"] = fo["gateway"]
            elif fi.get("gateway"):
                extra["gateway"] = fi["gateway"]
            if fo.get("_condition_raw"):
                extra["_condition_raw"] = fo["_condition_raw"]
            elif fi.get("_condition_raw"):
                extra["_condition_raw"] = fi["_condition_raw"]
        else:
            if fi.get("gateway"):
                extra["gateway"] = fi["gateway"]
            elif fo.get("gateway"):
                extra["gateway"] = fo["gateway"]
            if fi.get("_condition_raw"):
                extra["_condition_raw"] = fi["_condition_raw"]
            elif fo.get("_condition_raw"):
                extra["_condition_raw"] = fo["_condition_raw"]
        return extra

    def _collapse_intermediate_nodes(self, model: dict[str, Any]) -> None:
        # Même étape de nommage que les flux (après IDs → noms, avant slugify) ;
        # ``_sync_intermediate_event_names`` + slugify des ``ie`` dans ``_apply_slugify`` évitent le décalage.
        inter_names = {ie["name"] for ie in model.get("_intermediate_events", [])}
        if not inter_names:
            return

        seq_flows = [f for f in model["flows"] if f.get("type") == "sequence"]
        other = [f for f in model["flows"] if f.get("type") != "sequence"]
        changed = True
        while changed:
            changed = False
            for n in list(inter_names):
                inc = [f for f in seq_flows if f.get("to") == n]
                out = [f for f in seq_flows if f.get("from") == n]
                if not inc and not out:
                    continue
                if not inc or not out:
                    logger.warning(
                        "Événement intermédiaire « %s » : %d entrée(s), %d sortie(s) — "
                        "retrait des arcs incidents pour éviter un nœud fantôme.",
                        n,
                        len(inc),
                        len(out),
                    )
                    drop = {id(f) for f in inc + out}
                    seq_flows = [f for f in seq_flows if id(f) not in drop]
                    changed = True
                    break

                new_edges: list[dict[str, Any]] = []
                if len(inc) == 1 and len(out) == 1:
                    fi, fo = inc[0], out[0]
                    u, v = fi["from"], fo["to"]
                    newf: dict[str, Any] = {
                        "id": None,
                        "from": u,
                        "to": v,
                        "type": "sequence",
                    }
                    newf.update(self._merge_flow_attrs(fi, fo, prefer_outgoing=True))
                    new_edges.append(newf)
                elif len(out) == 1:
                    fo = out[0]
                    v = fo["to"]
                    for fi in inc:
                        u = fi["from"]
                        newf = {
                            "id": None,
                            "from": u,
                            "to": v,
                            "type": "sequence",
                        }
                        newf.update(self._merge_flow_attrs(fi, fo, prefer_outgoing=True))
                        new_edges.append(newf)
                elif len(inc) == 1:
                    fi = inc[0]
                    u = fi["from"]
                    for fo in out:
                        v = fo["to"]
                        newf = {
                            "id": None,
                            "from": u,
                            "to": v,
                            "type": "sequence",
                        }
                        newf.update(self._merge_flow_attrs(fi, fo, prefer_outgoing=True))
                        new_edges.append(newf)
                else:
                    for fi in inc:
                        for fo in out:
                            u, v = fi["from"], fo["to"]
                            newf = {
                                "id": None,
                                "from": u,
                                "to": v,
                                "type": "sequence",
                            }
                            newf.update(self._merge_flow_attrs(fi, fo, prefer_outgoing=True))
                            new_edges.append(newf)

                drop = {id(f) for f in inc + out}
                seq_flows = [f for f in seq_flows if id(f) not in drop]
                seq_flows.extend(new_edges)
                changed = True
                break

        still = {
            n
            for n in inter_names
            for f in seq_flows
            if f.get("from") == n or f.get("to") == n
        }
        if still:
            logger.warning(
                "Après repliement, nœuds intermédiaires encore référencés : %s",
                ", ".join(sorted(still)),
            )
        model["flows"] = seq_flows + other

    def _apply_slugify(self, model: dict[str, Any]) -> None:
        def sg(s: str) -> str:
            return slugify_label(s)

        for a in model["activities"]:
            a["name"] = sg(a["name"])
        for g in model["gateways"]:
            g["name"] = sg(g["name"])
        for ie in model.get("_intermediate_events", []):
            if ie.get("name"):
                ie["name"] = sg(ie["name"])

        for f in model["flows"]:
            if f.get("type") != "sequence":
                continue
            f["from"] = sg(f["from"])
            f["to"] = sg(f["to"])
            if f.get("gateway"):
                f["gateway"] = sg(f["gateway"])
            raw = f.pop("_condition_raw", None)
            if raw:
                c = _slugify_condition(raw)
                if c:
                    f["condition"] = c

    def _finalize_dataset_shape(self, model: dict[str, Any]) -> None:
        for a in model["activities"]:
            a.pop("id", None)
            a.pop("type", None)
        for g in model["gateways"]:
            g.pop("id", None)

        seq = []
        for f in model["flows"]:
            if f.get("type") != "sequence":
                continue
            fo: dict[str, Any] = {
                "from": f["from"],
                "to": f["to"],
                "type": "sequence",
            }
            if f.get("gateway"):
                fo["gateway"] = f["gateway"]
            if f.get("condition"):
                fo["condition"] = f["condition"]
            seq.append(fo)
        model["flows"] = seq
        model.pop("_intermediate_events", None)

    def convert_ids_to_names(self, model: dict[str, Any]) -> dict[str, Any]:
        """
        Compatibilité : si le modèle est encore en IDs (import JSON brut),
        résout les références. Les modèles produits par ``parse_file`` sont déjà finaux.
        """
        if not model.get("activities"):
            return model
        if any("id" in a for a in model["activities"] if isinstance(a, dict)):
            self._convert_ids_to_names(model)
            self._sync_intermediate_event_names(model)
        return model

    def process_directory(
        self, directory_path: str, output_dir: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Parcourt un dossier de fichiers BPMN / XML (comme l'ancienne API)."""
        return process_bpmn_directory(directory_path, output_dir)


def process_bpmn_directory(
    directory_path: str, output_dir: Optional[str] = None
) -> list[dict[str, Any]]:
    """Parcourt un dossier de fichiers BPMN / XML."""
    if not os.path.exists(directory_path):
        logger.error("Directory does not exist: %s", directory_path)
        return []

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files: list[str] = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".xml") or file.endswith(".bpmn"):
                xml_files.append(os.path.join(root, file))

    logger.info("Found %d BPMN files in %s", len(xml_files), directory_path)
    parser = BPMNParser()
    models: list[dict[str, Any]] = []
    for file_path in tqdm(xml_files, desc="Parsing BPMN files"):
        model = parser.parse_file(file_path)
        if model:
            models.append(model)
            if output_dir:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(model, f, indent=2, ensure_ascii=False)

    logger.info("Successfully parsed %d BPMN models", len(models))
    return models


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse BPMN XML → JSON (format dataset)")
    ap.add_argument("input", help="Fichier ou dossier BPMN / XML")
    ap.add_argument("--output", "-o", help="Fichier JSON ou dossier de sortie")
    args = ap.parse_args()

    p = BPMNParser()
    if os.path.isdir(args.input):
        out_dir = args.output if args.output else os.path.join(args.input, "json")
        n = len(p.process_directory(args.input, out_dir))
        print(f"OK — {n} modèle(s) → {out_dir}")
        return

    model = p.parse_file(args.input)
    if not model:
        print("Échec du parsing")
        return
    out = args.output
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2, ensure_ascii=False)
        print("Écrit :", out)
    else:
        print(json.dumps(model, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
