#!/usr/bin/env python3
"""
Automated Multi-Harness Setup Script for Ephemeral LoRA Proxy.
Configures OpenClaw, Hermes Agent, and Pi Coding Agent to route through
the local ephemeral weight consolidation proxy (http://127.0.0.1:8090/v1)
with harness-side context compaction safely deferred or disabled.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.error
import argparse
import difflib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


def get_default_proxy_url() -> str:
    """Reads proxy port from config.yaml if available, else defaults to 8090."""
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    config_path = repo_dir / "config.yaml"
    port = 8090
    if config_path.exists() and yaml is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "server" in data and "port" in data["server"]:
                    port = int(data["server"]["port"])
        except Exception:
            pass
    return f"http://127.0.0.1:{port}/v1"


def probe_proxy_models(proxy_url: str) -> Optional[List[str]]:
    """Probes GET <proxy_url>/models to verify connectivity and list available models."""
    url = f"{proxy_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ephemeral-harness-setup"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
                return models
    except Exception:
        pass
    return None


def create_backup(file_path: Path, dry_run: bool = False) -> Optional[Path]:
    """Creates a timestamped backup of the target file."""
    if not file_path.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.name}.bak.{timestamp}"
    if not dry_run:
        shutil.copy2(file_path, backup_path)
    return backup_path


def restore_backup(file_path: Path) -> bool:
    """Restores the most recent .bak.<timestamp> backup for a file."""
    pattern = f"{file_path.name}.bak.*"
    backups = sorted(file_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        print(f"  [Skip] No backup found for {file_path}")
        return False
    latest_backup = backups[0]
    shutil.copy2(latest_backup, file_path)
    print(f"  [Restored] {file_path} from {latest_backup.name}")
    return True


def show_diff(original_text: str, new_text: str, file_path: Path):
    """Prints a unified diff between original and planned file contents."""
    orig_lines = original_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, new_lines,
        fromfile=f"a/{file_path.name}",
        tofile=f"b/{file_path.name}"
    ))
    if diff:
        print(f"\n--- Planned Changes for {file_path} ---")
        sys.stdout.writelines(diff)
        print("----------------------------------------\n")
    else:
        print(f"  [No Change] {file_path} is already up to date.")


# -----------------------------------------------------------------------------
# 1. OpenClaw Configurator
# -----------------------------------------------------------------------------
def configure_openclaw(
    proxy_url: str,
    model_id: str,
    context_window: int,
    set_default: bool,
    dry_run: bool,
) -> bool:
    config_dir = Path.home() / ".openclaw"
    config_file = config_dir / "openclaw.json"
    print(f"\n[OpenClaw] Configuring {config_file}...")

    original_text = ""
    config_data: Dict[str, Any] = {}
    if config_file.exists():
        try:
            original_text = config_file.read_text(encoding="utf-8")
            config_data = json.loads(original_text)
        except Exception as e:
            print(f"  [Error] Failed to read existing {config_file}: {e}")
            return False
    else:
        config_data = {}

    # Deep-merge models.providers["ephemeral-local"]
    models = config_data.setdefault("models", {})
    providers = models.setdefault("providers", {})
    providers["ephemeral-local"] = {
        "baseUrl": proxy_url,
        "apiKey": "openclaw-session",
        "api": "openai-completions",
        "models": [
            {
                "id": model_id,
                "name": "Local Strix Halo (Ephemeral LoRA)",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": context_window,
                "maxTokens": 8192,
            }
        ],
    }

    # Allowlist in agents.defaults.models
    agents = config_data.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    agent_models = defaults.setdefault("models", {})
    agent_models[f"ephemeral-local/{model_id}"] = {"alias": "ephemeral"}

    if set_default:
        defaults.setdefault("model", {})["primary"] = f"ephemeral-local/{model_id}"
        print(f"  [Default] Set primary model to 'ephemeral-local/{model_id}'")

    new_text = json.dumps(config_data, indent=2) + "\n"

    if dry_run:
        show_diff(original_text, new_text, config_file)
        return True

    config_dir.mkdir(parents=True, exist_ok=True)
    backup_path = create_backup(config_file, dry_run=False)
    if backup_path:
        print(f"  [Backup] Created backup at {backup_path.name}")

    config_file.write_text(new_text, encoding="utf-8")
    print(f"  [Success] OpenClaw configured: provider 'ephemeral-local' with model '{model_id}'")
    return True


# -----------------------------------------------------------------------------
# 2. Hermes Agent Configurator
# -----------------------------------------------------------------------------
def configure_hermes(
    proxy_url: str,
    model_id: str,
    context_window: int,
    set_default: bool,
    dry_run: bool,
) -> bool:
    config_dir = Path.home() / ".hermes"
    config_file = config_dir / "config.yaml"
    env_file = config_dir / ".env"
    print(f"\n[Hermes Agent] Configuring {config_file} and {env_file}...")

    # Step A: Update .env with API key
    env_original = ""
    env_lines = []
    key_line = "EPHEMERAL_LORA_API_KEY=hermes-session"
    key_found = False

    if env_file.exists():
        env_original = env_file.read_text(encoding="utf-8")
        for line in env_original.splitlines():
            if line.startswith("EPHEMERAL_LORA_API_KEY="):
                env_lines.append(key_line)
                key_found = True
            else:
                env_lines.append(line)

    if not key_found:
        env_lines.append(key_line)
    env_new = "\n".join(env_lines) + "\n"

    # Step B: Update config.yaml preserving existing comments
    yaml_original = ""
    if config_file.exists():
        yaml_original = config_file.read_text(encoding="utf-8")

    # Prepare provider & alias snippets
    provider_block = (
        f"providers:\n"
        f"  ephemeral-local:\n"
        f"    base_url: \"{proxy_url}\"\n"
        f"    key_env: \"EPHEMERAL_LORA_API_KEY\"\n"
    )
    alias_block = (
        f"model_aliases:\n"
        f"  ephemeral:\n"
        f"    provider: \"custom\"\n"
        f"    base_url: \"{proxy_url}\"\n"
        f"    key_env: \"EPHEMERAL_LORA_API_KEY\"\n"
        f"    model: \"{model_id}\"\n"
        f"    context_length: {context_window}\n"
    )

    yaml_new = yaml_original
    if not yaml_new.strip():
        yaml_new = provider_block + "\n" + alias_block
    else:
        # Check providers
        if "ephemeral-local:" not in yaml_new:
            if "\nproviders:" in yaml_new:
                yaml_new = yaml_new.replace(
                    "\nproviders:",
                    f"\nproviders:\n  ephemeral-local:\n    base_url: \"{proxy_url}\"\n    key_env: \"EPHEMERAL_LORA_API_KEY\"",
                    1
                )
            else:
                yaml_new += f"\n\n# Ephemeral LoRA Provider\n{provider_block}"

        # Check model_aliases
        if "  ephemeral:" not in yaml_new:
            if "\nmodel_aliases:" in yaml_new:
                yaml_new = yaml_new.replace(
                    "\nmodel_aliases:",
                    f"\nmodel_aliases:\n  ephemeral:\n    provider: \"custom\"\n    base_url: \"{proxy_url}\"\n    key_env: \"EPHEMERAL_LORA_API_KEY\"\n    model: \"{model_id}\"\n    context_length: {context_window}",
                    1
                )
            else:
                yaml_new += f"\n# Ephemeral LoRA Model Alias\n{alias_block}"

    if set_default:
        # Replace or insert model: block
        default_block = (
            f"model:\n"
            f"  provider: custom\n"
            f"  base_url: \"{proxy_url}\"\n"
            f"  key_env: \"EPHEMERAL_LORA_API_KEY\"\n"
            f"  default: \"{model_id}\"\n"
            f"  context_length: {context_window}\n"
        )
        if "\nmodel:\n" in yaml_new or yaml_new.startswith("model:\n"):
            # Replace active model section up to next unindented block
            import re
            yaml_new = re.sub(r"(^|\n)model:\n(  [^\n]*\n)*", r"\1" + default_block, yaml_new, count=1)
        else:
            yaml_new = default_block + "\n" + yaml_new
        print(f"  [Default] Set Hermes default model to '{model_id}' (context: {context_window})")

    if dry_run:
        show_diff(env_original, env_new, env_file)
        show_diff(yaml_original, yaml_new, config_file)
        return True

    config_dir.mkdir(parents=True, exist_ok=True)
    bak_env = create_backup(env_file, dry_run=False)
    if bak_env:
        print(f"  [Backup] Created backup at {bak_env.name}")
    env_file.write_text(env_new, encoding="utf-8")

    bak_cfg = create_backup(config_file, dry_run=False)
    if bak_cfg:
        print(f"  [Backup] Created backup at {bak_cfg.name}")
    config_file.write_text(yaml_new, encoding="utf-8")

    print(f"  [Success] Hermes Agent configured: model_aliases.ephemeral -> '{model_id}'")
    return True


# -----------------------------------------------------------------------------
# 3. Pi Coding Agent Configurator
# -----------------------------------------------------------------------------
def configure_pi(
    proxy_url: str,
    model_id: str,
    context_window: int,
    set_default: bool,
    dry_run: bool,
) -> bool:
    config_dir = Path.home() / ".pi" / "agent"
    config_file = config_dir / "models.json"
    print(f"\n[Pi Coding Agent] Configuring {config_file}...")

    original_text = ""
    config_data: Dict[str, Any] = {}
    if config_file.exists():
        try:
            original_text = config_file.read_text(encoding="utf-8")
            config_data = json.loads(original_text)
        except Exception as e:
            print(f"  [Error] Failed to read {config_file}: {e}")
            return False

    providers = config_data.setdefault("providers", {})
    providers["ephemeral-local"] = {
        "baseUrl": proxy_url,
        "apiKey": "pi-session",
        "api": "openai-completions",
        "models": [
            {
                "id": model_id,
                "name": "Local Strix Halo (Ephemeral LoRA)",
                "contextWindow": context_window,
                "maxTokens": 8192,
            }
        ],
    }

    if set_default:
        config_data["defaultProvider"] = "ephemeral-local"
        config_data["defaultModel"] = model_id
        print(f"  [Default] Set Pi default model to 'ephemeral-local/{model_id}'")

    new_text = json.dumps(config_data, indent=2) + "\n"

    if dry_run:
        show_diff(original_text, new_text, config_file)
        return True

    config_dir.mkdir(parents=True, exist_ok=True)
    backup_path = create_backup(config_file, dry_run=False)
    if backup_path:
        print(f"  [Backup] Created backup at {backup_path.name}")

    config_file.write_text(new_text, encoding="utf-8")
    print(f"  [Success] Pi Coding Agent configured: provider 'ephemeral-local' with model '{model_id}'")
    print(f"  [Note] Configured contextWindow: {context_window}. This prevents Pi's built-in text compaction")
    print("         from triggering, delegating working-memory consolidation to ephemeral LoRA.")
    return True


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Automated setup tool for OpenClaw, Hermes Agent, and Pi with Ephemeral LoRA Proxy."
    )
    parser.add_argument(
        "--harnesses",
        default="all",
        help="Harnesses to configure: 'all' or comma-separated list 'openclaw,hermes,pi' (default: all)",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Proxy endpoint URL (default: http://127.0.0.1:<port>/v1 from config.yaml)",
    )
    parser.add_argument(
        "--model-id",
        default="ephemeral-local",
        help="Model ID to register in harnesses (default: ephemeral-local)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=131072,
        help="Advertised context window in tokens (default: 131072)",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Switch each harness's active default model to the local ephemeral proxy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned configuration changes and diffs without modifying files",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the most recent backups created by this tool",
    )

    args = parser.parse_args()

    # Restore workflow
    if args.restore:
        print("Restoring harness configurations from latest backups...")
        restore_backup(Path.home() / ".openclaw" / "openclaw.json")
        restore_backup(Path.home() / ".hermes" / "config.yaml")
        restore_backup(Path.home() / ".hermes" / ".env")
        restore_backup(Path.home() / ".pi" / "agent" / "models.json")
        print("Restore completed.")
        return

    proxy_url = args.proxy_url or get_default_proxy_url()
    model_id = args.model_id

    print("====================================================================")
    print(" Ephemeral LoRA Proxy — Agentic Harness Configurator               ")
    print("====================================================================")
    print(f"Target Proxy URL   : {proxy_url}")
    print(f"Model ID           : {model_id}")
    print(f"Advertised Window  : {args.context_window} tokens")
    print(f"Set as Default     : {args.set_default}")
    print(f"Dry Run Mode       : {args.dry_run}")

    # Auto-detect or parse harnesses
    requested = [h.strip().lower() for h in args.harnesses.split(",") if h.strip()]
    target_harnesses = []

    if "all" in requested:
        # Detect present harnesses or scaffold all if explicitly selected
        openclaw_present = (Path.home() / ".openclaw").exists() or bool(shutil.which("openclaw"))
        hermes_present = (Path.home() / ".hermes").exists() or bool(shutil.which("hermes"))
        pi_present = (Path.home() / ".pi").exists() or bool(shutil.which("pi"))

        if openclaw_present:
            target_harnesses.append("openclaw")
        if hermes_present:
            target_harnesses.append("hermes")
        if pi_present:
            target_harnesses.append("pi")

        if not target_harnesses:
            print("\n[Detection] No existing harness directories found (~/.openclaw, ~/.hermes, ~/.pi).")
            print("Configuring all 3 harnesses for immediate availability.")
            target_harnesses = ["openclaw", "hermes", "pi"]
    else:
        target_harnesses = requested

    print(f"\nConfiguring target harnesses: {', '.join(target_harnesses)}")

    for h in target_harnesses:
        if h == "openclaw":
            configure_openclaw(proxy_url, model_id, args.context_window, args.set_default, args.dry_run)
        elif h == "hermes":
            configure_hermes(proxy_url, model_id, args.context_window, args.set_default, args.dry_run)
        elif h == "pi":
            configure_pi(proxy_url, model_id, args.context_window, args.set_default, args.dry_run)
        else:
            print(f"  [Warning] Unknown harness '{h}', skipping.")

    # Probe live proxy status
    print("\n--------------------------------------------------------------------")
    print(" Post-Setup Verification & Live Status Check                        ")
    print("--------------------------------------------------------------------")
    models = probe_proxy_models(proxy_url)
    if models is not None:
        print(f"[ONLINE] Proxy is responding at {proxy_url}")
        print(f"         Available models: {', '.join(models)}")
    else:
        print(f"[OFFLINE] Proxy not responding at {proxy_url}")
        print("          Start it with: bash scripts/launch_stack.sh")

    print("\nVerification Commands:")
    if "openclaw" in target_harnesses:
        print("  - OpenClaw     : openclaw models status")
        print(f"                   openclaw chat --model ephemeral-local/{model_id}")
    if "hermes" in target_harnesses:
        print("  - Hermes Agent : hermes model")
        print("                   hermes chat -m ephemeral")
    if "pi" in target_harnesses:
        print("  - Pi Agent     : pi --list-models")
        print(f"                   pi --model ephemeral-local/{model_id}")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
