#!/usr/bin/env python3
"""
Interactive Source Tier Manager
Add, remove, and view sources in the AI domain tier configuration.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONFIG_FILE = Path(__file__).parent / "services" / "ai_domain_context.py"

# Tier names in order
TIER_NAMES = [
    "TIER_1_FOUNDATIONAL",
    "TIER_2_STRATEGIC",
    "TIER_3_POLICY",
    "TIER_4_PRACTITIONER",
]

TIER_DISPLAY = {
    "TIER_1_FOUNDATIONAL": "Tier 1: Foundational (Academic & Research Labs)",
    "TIER_2_STRATEGIC": "Tier 2: Strategic (Business Intelligence & Tech News)",
    "TIER_3_POLICY": "Tier 3: Policy (AI Governance & Think Tanks)",
    "TIER_4_PRACTITIONER": "Tier 4: Practitioner (Developer Blogs & Tutorials)",
}


def load_tiers() -> Tuple[Dict[str, List[str]], str]:
    """
    Parse AI_DOMAIN_TIERS from ai_domain_context.py.

    Returns:
        Tuple of (tiers dict, full file content)
    """
    content = CONFIG_FILE.read_text()

    tiers = {}
    for tier_name in TIER_NAMES:
        # Find the domains list for this tier
        pattern = rf'"{tier_name}":\s*\{{\s*"description":\s*"[^"]*",\s*"domains":\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            domains_str = match.group(1)
            # Extract domain strings, handling comments
            domain_pattern = r'"([^"]+)"'
            domains = re.findall(domain_pattern, domains_str)
            tiers[tier_name] = domains
        else:
            tiers[tier_name] = []

    return tiers, content


def display_tiers(tiers: Dict[str, List[str]]) -> None:
    """Pretty print all tiers with numbered domains."""
    print("\n" + "=" * 60)
    print("AI DOMAIN TIERS")
    print("=" * 60)

    for tier_name in TIER_NAMES:
        domains = tiers.get(tier_name, [])
        print(f"\n{TIER_DISPLAY[tier_name]}")
        print("-" * 50)

        if not domains:
            print("  (no domains)")
        else:
            for i, domain in enumerate(domains, 1):
                print(f"  {i:2}. {domain}")

        print(f"  Total: {len(domains)} domains")

    print("\n" + "=" * 60)


def find_domain(tiers: Dict[str, List[str]], domain: str) -> Optional[str]:
    """Find which tier contains a domain."""
    domain_lower = domain.lower()
    for tier_name, domains in tiers.items():
        for d in domains:
            if d.lower() == domain_lower:
                return tier_name
    return None


def add_domain(tiers: Dict[str, List[str]], tier_name: str, domain: str) -> bool:
    """
    Add domain to tier.

    Returns:
        True if added, False if duplicate
    """
    domain = domain.strip().lower()

    # Check for duplicates across all tiers
    existing_tier = find_domain(tiers, domain)
    if existing_tier:
        print(f"Domain '{domain}' already exists in {TIER_DISPLAY[existing_tier]}")
        return False

    tiers[tier_name].append(domain)
    print(f"Added '{domain}' to {TIER_DISPLAY[tier_name]}")
    return True


def remove_domain(tiers: Dict[str, List[str]], tier_name: str, domain: str) -> bool:
    """
    Remove domain from tier.

    Returns:
        True if removed, False if not found
    """
    domain_lower = domain.lower()
    for i, d in enumerate(tiers[tier_name]):
        if d.lower() == domain_lower:
            removed = tiers[tier_name].pop(i)
            print(f"Removed '{removed}' from {TIER_DISPLAY[tier_name]}")
            return True

    print(f"Domain '{domain}' not found in {TIER_DISPLAY[tier_name]}")
    return False


def move_domain(tiers: Dict[str, List[str]], domain: str, target_tier: str) -> bool:
    """
    Move domain from its current tier to target tier.

    Returns:
        True if moved, False otherwise
    """
    source_tier = find_domain(tiers, domain)

    if not source_tier:
        print(f"Domain '{domain}' not found in any tier")
        return False

    if source_tier == target_tier:
        print(f"Domain '{domain}' is already in {TIER_DISPLAY[target_tier]}")
        return False

    # Find and remove from source
    domain_lower = domain.lower()
    actual_domain = None
    for i, d in enumerate(tiers[source_tier]):
        if d.lower() == domain_lower:
            actual_domain = tiers[source_tier].pop(i)
            break

    # Add to target
    if actual_domain:
        tiers[target_tier].append(actual_domain)
        print(f"Moved '{actual_domain}' from {TIER_DISPLAY[source_tier]} to {TIER_DISPLAY[target_tier]}")
        return True

    return False


def save_tiers(tiers: Dict[str, List[str]]) -> bool:
    """
    Write updated AI_DOMAIN_TIERS back to file.

    Returns:
        True if saved successfully
    """
    content = CONFIG_FILE.read_text()

    for tier_name in TIER_NAMES:
        domains = tiers[tier_name]

        # Build the new domains list string with proper formatting
        if domains:
            # Group domains with comments based on original structure
            domains_str = "\n"
            for domain in domains:
                domains_str += f'            "{domain}",\n'
            domains_str += "        "
        else:
            domains_str = ""

        # Pattern to match the domains list for this tier
        pattern = rf'("{tier_name}":\s*\{{\s*"description":\s*"[^"]*",\s*"domains":\s*\[)[^\]]*(\])'
        replacement = rf'\g<1>{domains_str}\g<2>'

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    CONFIG_FILE.write_text(content)
    print(f"\nChanges saved to {CONFIG_FILE}")
    return True


def select_tier(prompt: str = "Select tier") -> Optional[str]:
    """Interactive tier selection."""
    print(f"\n{prompt}:")
    for i, tier_name in enumerate(TIER_NAMES, 1):
        print(f"  {i}. {TIER_DISPLAY[tier_name]}")

    try:
        choice = input("\nEnter number (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(TIER_NAMES):
            return TIER_NAMES[idx]
        print("Invalid selection")
    except ValueError:
        print("Invalid input")
    return None


def interactive_add(tiers: Dict[str, List[str]]) -> None:
    """Interactive domain addition."""
    tier_name = select_tier("Select tier to add domain to")
    if not tier_name:
        return

    domain = input("Enter domain to add (e.g., example.com): ").strip()
    if domain:
        add_domain(tiers, tier_name, domain)


def interactive_remove(tiers: Dict[str, List[str]]) -> None:
    """Interactive domain removal."""
    tier_name = select_tier("Select tier to remove domain from")
    if not tier_name:
        return

    # Show domains in selected tier
    domains = tiers[tier_name]
    if not domains:
        print(f"No domains in {TIER_DISPLAY[tier_name]}")
        return

    print(f"\nDomains in {TIER_DISPLAY[tier_name]}:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    try:
        choice = input("\nEnter number or domain name (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return

        # Try as number first
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(domains):
                remove_domain(tiers, tier_name, domains[idx])
                return
            print("Invalid selection")
        except ValueError:
            # Treat as domain name
            remove_domain(tiers, tier_name, choice)
    except Exception as e:
        print(f"Error: {e}")


def interactive_move(tiers: Dict[str, List[str]]) -> None:
    """Interactive domain move between tiers."""
    domain = input("Enter domain to move: ").strip()
    if not domain:
        return

    source_tier = find_domain(tiers, domain)
    if not source_tier:
        print(f"Domain '{domain}' not found in any tier")
        return

    print(f"\n'{domain}' is currently in {TIER_DISPLAY[source_tier]}")

    target_tier = select_tier("Select destination tier")
    if target_tier:
        move_domain(tiers, domain, target_tier)


def search_domain(tiers: Dict[str, List[str]]) -> None:
    """Search for a domain across all tiers."""
    query = input("Enter domain to search for: ").strip().lower()
    if not query:
        return

    found = []
    for tier_name in TIER_NAMES:
        for domain in tiers[tier_name]:
            if query in domain.lower():
                found.append((tier_name, domain))

    if found:
        print(f"\nFound {len(found)} matching domain(s):")
        for tier_name, domain in found:
            tier_short = tier_name.replace("TIER_", "T").replace("_", " ").title()
            print(f"  {domain} ({tier_short})")
    else:
        print(f"No domains matching '{query}' found")


def main():
    """Main interactive loop."""
    print("\n" + "=" * 60)
    print("   AI SOURCE TIER MANAGER")
    print("=" * 60)

    if not CONFIG_FILE.exists():
        print(f"Error: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    tiers, _ = load_tiers()
    unsaved_changes = False

    while True:
        print("\n--- Main Menu ---")
        print("1. View all tiers")
        print("2. Add domain to tier")
        print("3. Remove domain from tier")
        print("4. Move domain between tiers")
        print("5. Search for domain")
        print("6. Save changes & exit")
        print("7. Exit without saving")

        if unsaved_changes:
            print("\n  * You have unsaved changes *")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            display_tiers(tiers)
        elif choice == "2":
            interactive_add(tiers)
            unsaved_changes = True
        elif choice == "3":
            interactive_remove(tiers)
            unsaved_changes = True
        elif choice == "4":
            interactive_move(tiers)
            unsaved_changes = True
        elif choice == "5":
            search_domain(tiers)
        elif choice == "6":
            if unsaved_changes:
                save_tiers(tiers)
            else:
                print("No changes to save")
            print("Goodbye!")
            break
        elif choice == "7":
            if unsaved_changes:
                confirm = input("Discard unsaved changes? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
