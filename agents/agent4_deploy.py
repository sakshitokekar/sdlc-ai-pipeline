# Deploy Agent — takes the Docker image Agent 3 built and actually runs it
# on the local Minikube Kubernetes cluster: loads the image into Minikube's
# internal Docker, writes and applies K8s manifests, verifies the Pod comes
# up healthy, and reports the reachable URL back to Jira.
from state.pipeline_state import StateSDLC
from tools.jira_tools import add_jira_comment
from tools.kubernetes_tools import (
    check_minikube_available,
    load_image_into_minikube,
    write_manifests,
    apply_manifests,
    wait_for_pod_ready,
    get_service_access_info,
)
from tools import log_utils as log

REPO_PATH = "../sample_app"
APP_NAME = "sample-app"


def run_deploy_agent_node(state: StateSDLC) -> dict:
    jira_ticket = state["jira_ticket_details"]
    ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")
    build_results = state.get("build_results", {})

    # --- Step 0: Agent 4 only makes sense if Agent 3 actually produced an
    # image. If build_results is missing/failed, there's nothing to deploy. ---
    if not build_results.get("success"):
        log.warn("No successful build to deploy — skipping Agent 4.")
        return {
            "deploy_results": {
                "success": False,
                "reason_code": "NO_IMAGE_TO_DEPLOY"
            }
        }

    image_tag = build_results["image_tag"]

    # --- Step 1: Verify Minikube is actually running before doing anything ---
    minikube_status = check_minikube_available()
    if not minikube_status["available"]:
        error_detail = minikube_status.get("error", "Unknown error")
        log.error(f"Minikube unavailable: {minikube_status['reason_code']} — {error_detail}")

        comment = f"""Agent 4 (Deploy) could not run for {ticket_key}.

WHO: Agent 4 (Deploy Agent)
WHAT: Attempted to deploy {image_tag} to Minikube but the cluster is not available
WHY: {minikube_status['reason_code']}: {error_detail}
ACTION NEEDED: Run 'minikube start', then re-run the pipeline from this ticket
"""
        add_jira_comment(ticket_key, comment)
        return {
            "deploy_results": {
                "success": False,
                "reason_code": minikube_status["reason_code"],
                "error": error_detail
            }
        }

    # --- Step 2: Load the image (built on the host's Docker) into Minikube's
    # separate internal Docker daemon, since Kubernetes can only use images
    # that exist inside Minikube's own environment ---
    log.step(f"Loading {image_tag} into Minikube...")
    load_result = load_image_into_minikube(image_tag)
    if not load_result["success"]:
        log.error(f"Image load failed: {load_result['reason_code']}")
        log.full(f"Image load error for {ticket_key}:\n{load_result.get('error', '')}")
        comment = f"""Agent 4 (Deploy) FAILED for {ticket_key}.

WHO: Agent 4 (Deploy Agent)
WHAT: Attempted to load {image_tag} into Minikube
RESULT: {load_result['reason_code']}
"""
        add_jira_comment(ticket_key, comment)
        return {"deploy_results": load_result}
    log.success(f"Image loaded into Minikube: {image_tag}")

    # --- Step 3: Generate deterministic K8s manifests (Deployment + Service) ---
    manifest_paths = write_manifests(REPO_PATH, APP_NAME, image_tag, ticket_key)
    log.step(f"Wrote manifests: {manifest_paths['deployment_path']}, {manifest_paths['service_path']}")

    # --- Step 4: Apply manifests — idempotent, safe to re-run every time ---
    apply_result = apply_manifests(manifest_paths["deployment_path"], manifest_paths["service_path"])
    if not apply_result["success"]:
        log.error(f"kubectl apply failed: {apply_result['reason_code']}")
        log.full(f"kubectl apply output for {ticket_key}:\n{apply_result.get('output', '')}")
        comment = f"""Agent 4 (Deploy) FAILED for {ticket_key}.

WHO: Agent 4 (Deploy Agent)
WHAT: Attempted to apply Kubernetes manifests for {image_tag}
RESULT: {apply_result['reason_code']}
OUTPUT (tail): {apply_result.get('output', '')[-500:]}
"""
        add_jira_comment(ticket_key, comment)
        return {"deploy_results": apply_result}
    log.success("Manifests applied")

    # --- Step 5: Wait and verify the Pod actually reaches Ready state —
    # a successfully applied manifest doesn't guarantee the Pod is healthy ---
    log.step("Waiting for Pod to become ready...")
    readiness = wait_for_pod_ready(APP_NAME)
    if not readiness["ready"]:
        log.error(f"Pod did not become ready: {readiness['reason_code']} (last status: {readiness.get('last_status', 'unknown')})")
        comment = f"""Agent 4 (Deploy) FAILED for {ticket_key}.

WHO: Agent 4 (Deploy Agent)
WHAT: Deployed {image_tag} but the Pod never reached Ready state
RESULT: {readiness['reason_code']} (last observed status: {readiness.get('last_status', 'unknown')})
ACTION NEEDED: Run 'kubectl describe pod -l app={APP_NAME}' to diagnose
"""
        add_jira_comment(ticket_key, comment)
        return {"deploy_results": {"success": False, "reason_code": readiness["reason_code"]}}
    log.success(f"Pod ready: {readiness['pod_name']}")

    # --- Step 6: Resolve access info for this deployment. minikube's own
    # --url command opens a blocking tunnel on macOS/Docker-driver, so we
    # report the NodePort plus the exact command to open a live tunnel,
    # rather than an unreliable captured URL. ---
    access_info = get_service_access_info(APP_NAME)
    node_port = access_info.get("node_port", "unknown")
    tunnel_command = access_info.get("manual_tunnel_command", f"minikube service {APP_NAME}-service --url")

    # --- Step 7: Update Jira with audit trail. Status still not advanced
    # to "Done" here — that's Agent 6's job once we build it, since a
    # successful Minikube deploy is a staging step, not final production. ---
    comment = f"""Agent 4 (Deploy) completed for {ticket_key}.

WHO: Agent 4 (Deploy Agent)
WHAT: Deployed {image_tag} to local Minikube cluster
WHY: Verify the built image runs correctly in a real Kubernetes environment before production promotion
POD: {readiness['pod_name']}
NODEPORT: {node_port}
TO ACCESS: run `{tunnel_command}` in a terminal to open a live tunnel and get the URL
RESULT: DEPLOY_SUCCESS
"""
    add_jira_comment(ticket_key, comment)

    return {
        "deploy_results": {
            "success": True,
            "pod_name": readiness["pod_name"],
            "node_port": node_port,
            "tunnel_command": tunnel_command,
            "image_tag": image_tag,
            "reason_code": "DEPLOY_SUCCESS"
        }
    }


if __name__ == "__main__":
    test_state = {
        "user_input": "test",
        "jira_ticket_details": {"ticket_key": "SDLC-9", "url": "https://sakshitokekar.atlassian.net/browse/SDLC-9"},
        "code": "",
        "test_results": {},
        "dev_test_retry_count": 0,
        "human_decision": "",
        "build_results": {"success": True, "image_tag": "sample-app:493fc5f", "reason_code": "BUILD_SUCCESS"},
        "deploy_results": {}
    }
    result = run_deploy_agent_node(test_state)
    print(result)