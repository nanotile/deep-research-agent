#!/usr/bin/env python3
"""
Complete Firewall and Port Analysis Tool

Uses gcloud to check actual firewall rules and compares with listening services.
"""

import subprocess
import json
import sys
from typing import Dict, List, Set
from vm_ip_utils import get_vm_ip


def run_command(cmd: List[str], shell: bool = False) -> Dict:
    """Run a command and return result"""
    try:
        if shell:
            result = subprocess.run(
                ' '.join(cmd) if isinstance(cmd, list) else cmd,
                shell=True,
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_vm_info() -> Dict:
    """Get current VM information"""
    try:
        ip = get_vm_ip()
        # Get instance name from metadata
        result = run_command([
            'curl', '-s',
            'http://metadata.google.internal/computeMetadata/v1/instance/name',
            '-H', 'Metadata-Flavor: Google'
        ])

        instance_name = result['stdout'] if result['success'] else 'unknown'

        # Get zone from metadata
        result = run_command([
            'curl', '-s',
            'http://metadata.google.internal/computeMetadata/v1/instance/zone',
            '-H', 'Metadata-Flavor: Google'
        ])

        zone = result['stdout'].split('/')[-1] if result['success'] else 'unknown'

        return {
            'ip': ip,
            'instance': instance_name,
            'zone': zone
        }
    except Exception as e:
        return {'error': str(e)}


def get_vm_tags(instance_name: str, zone: str) -> List[str]:
    """Get network tags for the VM instance"""
    result = run_command([
        'gcloud', 'compute', 'instances', 'describe',
        instance_name,
        f'--zone={zone}',
        '--format=json'
    ])

    if result['success']:
        try:
            data = json.loads(result['stdout'])
            return data.get('tags', {}).get('items', [])
        except json.JSONDecodeError:
            return []
    return []


def get_firewall_rules() -> List[Dict]:
    """Get all firewall rules"""
    result = run_command([
        'gcloud', 'compute', 'firewall-rules', 'list',
        '--format=json'
    ])

    if result['success']:
        try:
            return json.loads(result['stdout'])
        except json.JSONDecodeError:
            return []
    return []


def parse_allowed_ports(rule: Dict) -> Set[int]:
    """Extract allowed ports from a firewall rule"""
    ports = set()

    for allow in rule.get('allowed', []):
        if 'ports' in allow:
            for port_spec in allow['ports']:
                if '-' in port_spec:
                    # Port range
                    start, end = port_spec.split('-')
                    ports.update(range(int(start), int(end) + 1))
                else:
                    # Single port
                    ports.add(int(port_spec))
        else:
            # No ports specified means all ports for the protocol
            pass

    return ports


def get_listening_ports() -> Dict[int, Dict]:
    """Get all listening ports and their bind addresses"""
    result = run_command('ss -tlnp 2>/dev/null | grep LISTEN', shell=True)

    ports = {}
    if result['success']:
        for line in result['stdout'].split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                local_addr = parts[3]

                # Parse address and port
                if ']:' in local_addr:
                    # IPv6
                    addr, port = local_addr.rsplit(']:', 1)
                    addr = addr.lstrip('[')
                elif ':' in local_addr:
                    # IPv4
                    addr, port = local_addr.rsplit(':', 1)
                else:
                    continue

                try:
                    port = int(port)

                    # Determine bind type
                    if addr in ('0.0.0.0', '::'):
                        bind_type = 'external'
                    elif addr.startswith('127.') or addr == '::1':
                        bind_type = 'localhost'
                    else:
                        bind_type = 'specific'

                    # Extract process info if available
                    process = None
                    for part in parts:
                        if 'users:' in part or 'pid=' in part:
                            process = part
                            break

                    ports[port] = {
                        'address': addr,
                        'bind_type': bind_type,
                        'process': process
                    }
                except ValueError:
                    continue

    return ports


def main():
    """Main analysis function"""
    print("="*80)
    print("🔥 COMPLETE FIREWALL & PORT ANALYSIS")
    print("="*80)

    # Get VM info
    print("\n📍 VM Information:")
    vm_info = get_vm_info()
    print(f"   Instance: {vm_info.get('instance', 'unknown')}")
    print(f"   Zone: {vm_info.get('zone', 'unknown')}")
    print(f"   External IP: {vm_info.get('ip', 'unknown')}")

    # Get VM tags
    instance_name = vm_info.get('instance', '')
    zone = vm_info.get('zone', '')

    if instance_name != 'unknown':
        tags = get_vm_tags(instance_name, zone)
        print(f"   Network Tags: {', '.join(tags) if tags else 'none'}")
    else:
        tags = []
        print("   Network Tags: unable to retrieve")

    # Get firewall rules
    print("\n🛡️  Firewall Rules Analysis:")
    print("-"*80)

    firewall_rules = get_firewall_rules()

    if not firewall_rules:
        print("   ❌ Unable to retrieve firewall rules (gcloud not configured?)")
    else:
        # Find rules that apply to this VM
        applicable_rules = []
        allowed_ports = set()

        for rule in firewall_rules:
            target_tags = rule.get('targetTags', [])

            # Rule applies if: no target tags OR VM has matching tag
            if not target_tags or any(tag in tags for tag in target_tags):
                applicable_rules.append(rule)

                # Extract ports
                ports = parse_allowed_ports(rule)
                allowed_ports.update(ports)

        print(f"\n   Found {len(applicable_rules)} rules applying to your VM:")
        for rule in applicable_rules:
            name = rule['name']
            allowed = rule.get('allowed', [])
            target_tags = rule.get('targetTags', [])

            print(f"\n   📋 {name}")

            # Show allowed protocols and ports
            for allow in allowed:
                protocol = allow.get('IPProtocol', 'unknown')
                ports = allow.get('ports', ['all'])
                print(f"      Protocol: {protocol}")
                print(f"      Ports: {', '.join(ports)}")

            if target_tags:
                print(f"      Target Tags: {', '.join(target_tags)}")
            else:
                print(f"      Target Tags: all instances")

        print(f"\n   📊 Summary of Allowed Ports:")
        if allowed_ports:
            sorted_ports = sorted(allowed_ports)
            # Group consecutive ports
            ranges = []
            start = sorted_ports[0]
            end = start

            for port in sorted_ports[1:] + [None]:
                if port == end + 1:
                    end = port
                else:
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    if port:
                        start = end = port

            print(f"      {', '.join(ranges[:20])}")
            if len(ranges) > 20:
                print(f"      ... and {len(ranges) - 20} more")
        else:
            print("      No specific ports found")

    # Get listening ports
    print("\n🔌 Currently Listening Services:")
    print("-"*80)

    listening = get_listening_ports()

    # Separate by bind type
    external = {p: i for p, i in listening.items() if i['bind_type'] == 'external'}
    localhost = {p: i for p, i in listening.items() if i['bind_type'] == 'localhost'}

    print(f"\n   ✓ Services Bound to 0.0.0.0 (ACCESSIBLE EXTERNALLY):")
    if external:
        for port, info in sorted(external.items()):
            accessible = "✓ ACCESSIBLE" if port in allowed_ports or not firewall_rules else "? Check firewall"
            print(f"      Port {port:5d}: {info['address']:15s} {accessible}")
            print(f"                  → http://{vm_info.get('ip', 'unknown')}:{port}")
    else:
        print("      (none)")

    print(f"\n   ❌ Services Bound to 127.0.0.1 (NOT ACCESSIBLE EXTERNALLY):")
    if localhost:
        for port, info in sorted(localhost.items()):
            print(f"      Port {port:5d}: {info['address']:15s} (localhost only)")
    else:
        print("      (none)")

    # Recommendations
    print("\n💡 Recommendations:")
    print("-"*80)

    if not external or len(external) <= 2:
        print("   ⚠️  Very few services are accessible externally!")
        print("   → Make sure your services bind to 0.0.0.0 (not 127.0.0.1)")
        print("   → Examples:")
        print("      Flask: app.run(host='0.0.0.0', port=PORT)")
        print("      Gradio: demo.launch(server_name='0.0.0.0', server_port=PORT)")
        print("      Node: app.listen(PORT, '0.0.0.0')")

    if localhost and len(localhost) > len(external):
        print(f"\n   📊 You have {len(localhost)} services bound to localhost")
        print("   → These won't be accessible from external networks")
        print("   → Change bind address from 127.0.0.1 to 0.0.0.0 if you need external access")

    if firewall_rules and external:
        blocked = [p for p in external.keys() if p not in allowed_ports]
        if blocked:
            print(f"\n   ⚠️  These ports are listening but BLOCKED by firewall:")
            for port in sorted(blocked):
                print(f"      Port {port}")
            print("\n   → Create firewall rules to allow these ports:")
            for port in sorted(blocked):
                print(f"      gcloud compute firewall-rules create allow-port-{port} \\")
                print(f"          --allow tcp:{port} --source-ranges 0.0.0.0/0 \\")
                print(f"          --target-tags {','.join(tags) if tags else 'your-tag'}")

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
