import json
import os
import re
import csv

def clean_alias(alias_str):
    if not alias_str or alias_str in ["—", "–", "Enter your input for this section here.", "Not available", ""]:
        return []
    # Remove markdown links like [(Author Year)](url)
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', alias_str)
    # Remove citations like [[Author Year]]
    cleaned = re.sub(r'\[\[.*?\]\]', '', cleaned)
    # Extract specific alias names if it says "This pattern has also been referred to as X or Y"
    if "referred to as" in cleaned:
        parts = cleaned.split("referred to as")[1].split("by")[0].strip()
        parts = parts.replace(" or ", ", ")
        aliases = [p.strip(" .") for p in parts.split(",")]
        return [a for a in aliases if a]
    elif "another alias for this pattern is" in cleaned:
        parts = cleaned.split("another alias for this pattern is")[1].strip()
        aliases = [p.strip(" .") for p in parts.split(",")]
        return [a for a in aliases if a]
    else:
        # Just comma separated
        return [a.strip() for a in cleaned.split(",") if a.strip()]

with open('/home/neilson/projects/qpa/data/quantum_patterns.json', 'r') as f:
    patterns_data = json.load(f)

patterns = []
for p in patterns_data:
    name = p['name']
    aliases = clean_alias(p.get('alias', ''))
    
    # Generate search terms
    search_terms = [name.lower()]
    for a in aliases:
         search_terms.append(a.lower())
    
    # Add acronyms from names like "Quantum Kernel Estimator (QKE)" -> "qke", "quantum kernel estimator"
    acronym_match = re.search(r'\(([^)]+)\)', name)
    if acronym_match:
        acronym = acronym_match.group(1).lower()
        search_terms.append(acronym)
        search_terms[0] = name.replace(f"({acronym_match.group(1)})", "").strip().lower()
        
    # Clean up terms
    clean_terms = []
    for term in search_terms:
         term = term.replace("-", " ")
         # Add version with dashes if it had spaces
         clean_terms.append(term)
         if " " in term:
             clean_terms.append(term.replace(" ", ""))
             clean_terms.append(term.replace(" ", "_"))
             
    # Deduplicate
    patterns.append({
        'name': name,
        'terms': list(set(clean_terms))
    })

# Now walk through the repository
repo_path = "/home/neilson/projects/qpa/target_github_projects/qiskit-algorithms/qiskit_algorithms"
matches = []

for root, dirs, files in os.walk(repo_path):
    for file in files:
        if not file.endswith('.py'):
            continue
            
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, "/home/neilson/projects/qpa/target_github_projects/qiskit-algorithms")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
        except Exception:
            continue
            
        file_name_lower = file.lower()
        
        # Check against patterns
        file_matches = set()
        for p in patterns:
            matched = False
            for term in p['terms']:
                # Skip overly broad terms if they are too short
                if len(term) < 3:
                     continue
                if term in file_name_lower or term in content:
                    matched = True
                    break
            if matched:
                file_matches.add(p['name'])
                
        if file_matches:
            for match in file_matches:
                matches.append({'File Name': rel_path, 'Pattern Match': match})

# Write to CSV
csv_path = "/home/neilson/projects/qpa/pattern_matches.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['File Name', 'Pattern Match'])
    writer.writeheader()
    for match in matches:
        writer.writerow(match)

print(f"Wrote {len(matches)} matches to {csv_path}")
print("Sample matches:")
for m in matches[:10]:
    print(m)
