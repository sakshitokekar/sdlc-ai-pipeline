# Kubernetes tools — checking Minikube availability, loading Docker images
# into Minikube's cluster, applying Deployment/Service manifests, and
# verifying the resulting Pod actually comes up healthy. Manifests are
# deterministic templates (not LLM-generated), same design decision as
# docker_tools.py's Dockerfile — infrastructure should be reproducible.
import subprocess
import time
from pathlib import Path


#     check_minikube_available() -> dict:
#         - verifies Minikube is installed AND its cluster is currently running
#         - returns {"available": bool, "reason_code": str, "error": str (if failed)}
#         - reason codes: MINIKUBE_AVAILABLE, MINIKUBE_NOT_INSTALLED, MINIKUBE_NOT_RUNNING
def check_minikube_available() -> dict:
    try:
        result = subprocess.run(
            ["minikube", "status"],
            capture_output=True, text=True, timeout=15
        )
        # minikube status returns non-zero if the cluster isn't running,
        # even though the CLI itself is installed and responded correctly
        if result.returncode == 0 and "Running" in result.stdout:
            return {"available": True, "reason_code": "MINIKUBE_AVAILABLE"}
        return {
            "available": False,
            "error": result.stdout + result.stderr,
            "reason_code": "MINIKUBE_NOT_RUNNING"
        }
    except FileNotFoundError:
        return {"available": False, "error": "minikube CLI not found on PATH", "reason_code": "MINIKUBE_NOT_INSTALLED"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "minikube status timed out", "reason_code": "MINIKUBE_TIMEOUT"}


#     load_image_into_minikube(image_tag) -> dict:
#         - copies an already-built Docker image (built by Agent 3 on the
#           host's Docker daemon) into Minikube's separate internal Docker
#           daemon, since the two are isolated environments by default
#         - this is the local-development equivalent of pushing to a real
#           image registry, which is what a real cloud Kubernetes deployment
#           would use instead
#         - returns {"success": bool, "reason_code": str}
def load_image_into_minikube(image_tag: str) -> dict:
    try:
        result = subprocess.run(
            ["minikube", "image", "load", image_tag],
            capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Image load timed out after 180s", "reason_code": "IMAGE_LOAD_TIMEOUT", "recoverable": True}

    if result.returncode == 0:
        return {"success": True, "reason_code": "IMAGE_LOADED"}
    return {
        "success": False,
        "error": result.stdout + result.stderr,
        "reason_code": "IMAGE_LOAD_FAILED",
        "recoverable": True
    }


# Deterministic Kubernetes manifest templates. {image_tag}, {app_name}, and
# {ticket_key} are substituted in at generation time. Two separate objects:
#   Deployment — keeps N Pods of the app running, restarts them if they crash
#   Service    — stable network address routing traffic to those Pods
_DEPLOYMENT_TEMPLATE = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
    ticket: {ticket_key}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
        - name: {app_name}
          image: {image_tag}
          imagePullPolicy: Never
          ports:
            - containerPort: 5000
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"
"""

_SERVICE_TEMPLATE = """apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
  labels:
    app: {app_name}
spec:
  type: NodePort
  selector:
    app: {app_name}
  ports:
    - port: 5000
      targetPort: 5000
"""


#     write_manifests(repo_path, app_name, image_tag, ticket_key) -> dict:
#         - writes deployment.yaml and service.yaml to a k8s/ folder in the repo
#         - imagePullPolicy: Never tells Kubernetes "only use the image if
#           it's already loaded locally — never try to pull from a registry",
#           which is correct for our local-image-load workflow and also
#           fails fast with a clear error if the image load step was skipped
#         - overwrites any existing manifests each run, since they're
#           generated artifacts, not something meant to be hand-edited
#         - returns {"deployment_path": str, "service_path": str}
def write_manifests(repo_path: str, app_name: str, image_tag: str, ticket_key: str) -> dict:
    k8s_dir = Path(repo_path) / "k8s"
    k8s_dir.mkdir(exist_ok=True)

    deployment_path = k8s_dir / "deployment.yaml"
    service_path = k8s_dir / "service.yaml"

    deployment_path.write_text(_DEPLOYMENT_TEMPLATE.format(
        app_name=app_name, image_tag=image_tag, ticket_key=ticket_key
    ))
    service_path.write_text(_SERVICE_TEMPLATE.format(app_name=app_name))

    return {
        "deployment_path": str(deployment_path),
        "service_path": str(service_path)
    }


#     apply_manifests(deployment_path, service_path) -> dict:
#         - runs `kubectl apply -f` for both manifests
#         - kubectl apply is idempotent: safe to run repeatedly, it will
#           create resources that don't exist and update ones that do,
#           rather than erroring on "already exists"
#         - returns {"success": bool, "output": str, "reason_code": str}
def apply_manifests(deployment_path: str, service_path: str) -> dict:
    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", deployment_path, "-f", service_path],
            capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "kubectl apply timed out", "reason_code": "APPLY_TIMEOUT", "recoverable": True}

    output = result.stdout + result.stderr
    if result.returncode == 0:
        return {"success": True, "output": output, "reason_code": "MANIFESTS_APPLIED"}
    return {"success": False, "output": output, "reason_code": "APPLY_FAILED", "recoverable": True}


#     wait_for_pod_ready(app_name, timeout_seconds=60) -> dict:
#         - polls kubectl to confirm the Deployment's Pod actually reaches
#           Ready state, not just that the manifest was accepted
#         - a Pod can be "applied" successfully but still fail to actually
#           start (bad image, crash loop, resource limits) — this catches
#           that gap between "kubectl accepted the YAML" and "it's really running"
#         - returns {"ready": bool, "pod_name": str, "reason_code": str}
def wait_for_pod_ready(app_name: str, timeout_seconds: int = 60) -> dict:
    deadline = time.time() + timeout_seconds
    last_status = ""

    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", f"app={app_name}",
             "-o", "jsonpath={.items[0].metadata.name} {.items[0].status.phase}"],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        if output:
            parts = output.split(" ", 1)
            pod_name = parts[0]
            phase = parts[1] if len(parts) > 1 else "Unknown"
            last_status = phase

            if phase == "Running":
                # Also check the container's own readiness probe status,
                # not just the Pod phase — a Pod can be "Running" while
                # its container is still crash-looping
                ready_check = subprocess.run(
                    ["kubectl", "get", "pods", "-l", f"app={app_name}",
                     "-o", "jsonpath={.items[0].status.containerStatuses[0].ready}"],
                    capture_output=True, text=True
                )
                if ready_check.stdout.strip() == "true":
                    return {"ready": True, "pod_name": pod_name, "reason_code": "POD_READY"}

        time.sleep(2)

    return {"ready": False, "pod_name": None, "reason_code": "POD_NOT_READY_TIMEOUT", "last_status": last_status}


#     get_service_access_info(app_name, node_port_command="kubectl get svc ...") -> dict:
#         - `minikube service --url` opens a persistent foreground tunnel on
#           macOS with the Docker driver (it never cleanly "finishes" the
#           way subprocess.run() expects) — unreliable to call this way
#         - instead, we report the NodePort directly via `kubectl get svc`,
#           plus the exact command the user can run themselves to open a
#           live tunnel when they actually want to browse to the app
#         - returns {"node_port": str, "manual_tunnel_command": str, "reason_code": str}
def get_service_access_info(app_name: str) -> dict:
    result = subprocess.run(
        ["kubectl", "get", "svc", f"{app_name}-service",
         "-o", "jsonpath={.spec.ports[0].nodePort}"],
        capture_output=True, text=True, timeout=15
    )
    node_port = result.stdout.strip()

    if result.returncode != 0 or not node_port:
        return {"node_port": None, "manual_tunnel_command": None, "reason_code": "SERVICE_INFO_FAILED", "error": result.stderr}

    tunnel_command = f"minikube service {app_name}-service --url"
    return {
        "node_port": node_port,
        "manual_tunnel_command": tunnel_command,
        "reason_code": "SERVICE_INFO_RESOLVED"
    }