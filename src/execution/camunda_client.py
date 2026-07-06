"""
Client REST Camunda 7 (engine-rest) et point d'entrée Camunda 8 (Orchestration API).

Sur Azure AKS, déployer Camunda via Helm (voir ``deploy/camunda-aks/``) puis
configurer ``CAMUNDA_REST_URL`` vers le service (Ingress ou port-forward).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class CamundaClientError(Exception):
    pass


class CamundaRestClient:
    """Adaptateur minimal Camunda 7 ``/engine-rest``."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        auth: Optional[tuple[str, str]] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("CAMUNDA_REST_URL") or "http://localhost:8080/engine-rest"
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.auth = auth
        if auth is None:
            user = os.environ.get("CAMUNDA_USER")
            pwd = os.environ.get("CAMUNDA_PASSWORD")
            if user and pwd:
                self.auth = (user, pwd)

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_s,
            auth=self.auth,
        )

    def health(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/version")
                return r.status_code == 200
        except Exception as e:
            logger.debug("Camunda health check failed: %s", e)
            return False

    def deploy_bpmn(self, bpmn_path: str | Path, deployment_name: str) -> dict[str, Any]:
        path = Path(bpmn_path)
        if not path.is_file():
            raise CamundaClientError(f"BPMN introuvable : {path}")
        with self._client() as c:
            with open(path, "rb") as f:
                files = {path.name: (path.name, f, "application/xml")}
                data = {
                    "deployment-name": deployment_name,
                    "enable-duplicate-filtering": "true",
                    "deploy-changed-only": "true",
                }
                r = c.post("/deployment/create", data=data, files=files)
        if r.status_code >= 400:
            raise CamundaClientError(f"Déploiement échoué ({r.status_code}): {r.text}")
        return r.json()

    def start_process(
        self,
        process_definition_key: str,
        *,
        variables: Optional[dict[str, Any]] = None,
        business_key: Optional[str] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if business_key:
            body["businessKey"] = business_key
        if variables:
            body["variables"] = {
                k: {"value": v, "type": _camunda_var_type(v)}
                for k, v in variables.items()
            }
        with self._client() as c:
            r = c.post(
                f"/process-definition/key/{process_definition_key}/start",
                json=body,
            )
        if r.status_code >= 400:
            raise CamundaClientError(f"Démarrage processus échoué ({r.status_code}): {r.text}")
        return r.json()

    def list_tasks(self, process_instance_id: Optional[str] = None) -> list[dict]:
        params: dict[str, str] = {}
        if process_instance_id:
            params["processInstanceId"] = process_instance_id
        with self._client() as c:
            r = c.get("/task", params=params)
        if r.status_code >= 400:
            raise CamundaClientError(f"Liste tâches échouée ({r.status_code}): {r.text}")
        return r.json()

    def complete_task(
        self,
        task_id: str,
        *,
        variables: Optional[dict[str, Any]] = None,
    ) -> None:
        body: dict[str, Any] = {}
        if variables:
            body["variables"] = {
                k: {"value": v, "type": _camunda_var_type(v)}
                for k, v in variables.items()
            }
        with self._client() as c:
            r = c.post(f"/task/{task_id}/complete", json=body or None)
        if r.status_code >= 400:
            raise CamundaClientError(f"Complétion tâche échouée ({r.status_code}): {r.text}")


def _camunda_var_type(value: Any) -> str:
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Integer"
    if isinstance(value, float):
        return "Double"
    return "String"


class Camunda8GatewayClient:
    """
    Camunda 8 — API REST Orchestration (Zeebe gateway proxy).

    Nécessite ``CAMUNDA8_REST_URL`` et token OAuth (``CAMUNDA8_TOKEN``) selon déploiement AKS.
    """

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        self.base_url = (
            base_url
            or os.environ.get("CAMUNDA8_REST_URL")
            or "http://localhost:8088/v2"
        ).rstrip("/")
        self.token = token or os.environ.get("CAMUNDA8_TOKEN", "")

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as c:
                r = c.get(f"{self.base_url}/topology", headers=self._headers())
                return r.status_code == 200
        except Exception:
            return False
