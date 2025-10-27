import json
import textwrap
import time

import requests
from bs4 import BeautifulSoup

from src.conf import config

PROJECT_ROOT = config.PROJECT_ROOT
QUANTUM_PATTERNS_REFERENCE_FILE = config.RESULTS_DIR / "quantum_patterns.json"


def get_all_pattern_summaries():
    """
    Fetches the master list of all quantum patterns from the API endpoint.
    This replaces the need for a hardcoded list.
    """
    master_list_url = "https://patternatlas.planqk.de/patternatlas/patternLanguages/af7780d5-1f97-4536-8da7-4194b093ab1d/patterns/"
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    print("Fetching master list of all available patterns...")
    try:
        response = requests.get(master_list_url, headers=headers, timeout=20)
        response.raise_for_status()  # Will raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # The pattern data is nested inside '_embedded' and 'patternModels' keys
        patterns_list = data["_embedded"]["patternModels"]
        print(
            f"Successfully found {len(patterns_list)} patterns available from the API."
        )
        return patterns_list

    except requests.exceptions.RequestException as e:
        print(f"[FATAL ERROR] Could not fetch the master list of patterns. Error: {e}")
    except KeyError:
        print(
            "[FATAL ERROR] The JSON structure of the master list seems to have changed. Could not find '_embedded' or 'patternModels'."
        )

    return []  # Return an empty list on failure


def download_quantum_pattern_details(pattern_summaries):
    """
    Takes a list of pattern summaries and downloads the detailed, rendered content for each one.
    """
    base_url = "https://patternatlas.planqk.de/patternatlas"
    all_patterns_data = []

    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    }

    total_patterns = len(pattern_summaries)
    print(f"\nAttempting to download details for {total_patterns} patterns.")

    for i, pattern_summary in enumerate(pattern_summaries, 1):
        # Note: The new JSON uses 'name', the old one used 'title'
        pattern_id = pattern_summary.get("id")
        pattern_name = pattern_summary.get("name")
        language_id = pattern_summary.get("patternLanguageId")

        if not all([pattern_id, pattern_name, language_id]):
            print(f"Skipping invalid entry: {pattern_summary}")
            continue

        print(f"({i}/{total_patterns}) Downloading: {pattern_name}")

        rendered_content_url = f"{base_url}/patternLanguages/{language_id}/patterns/{pattern_id}/renderedContent"
        headers["referer"] = (
            f"https://patternatlas.planqk.de/pattern-languages/{language_id}/{pattern_id}"
        )

        try:
            # A small delay is polite to the server
            time.sleep(0.2)
            response = requests.get(rendered_content_url, headers=headers, timeout=20)
            response.raise_for_status()
            pattern_details = response.json()

            content = pattern_details.get("renderedContent", {})
            if not content:
                print(f"  [Info] Pattern '{pattern_name}' has no rendered content.")
                continue

            extracted_content = {"name": pattern_name}
            sections_to_extract = [
                "Intent",
                "Alias",
                "Context",
                "Forces",
                "Solution",
                "Result",
            ]

            for section in sections_to_extract:
                raw_html = content.get(section, "Not available")
                soup = BeautifulSoup(raw_html, "html.parser")
                clean_text = soup.get_text(separator=" ", strip=True)
                extracted_content[section.lower()] = clean_text

            all_patterns_data.append(extracted_content)

        except requests.exceptions.RequestException as e:
            print(
                f"  [ERROR] Could not fetch details for {pattern_name}. URL: {rendered_content_url}. Error: {e}"
            )

    return all_patterns_data


if __name__ == "__main__":
    # 1. Dynamically get the list of patterns to process
    patterns_to_process = get_all_pattern_summaries()

    if patterns_to_process:
        # 2. Download the details for each pattern found
        detailed_patterns = download_quantum_pattern_details(patterns_to_process)

        if detailed_patterns:
            print("\n\n" + "=" * 80)
            print("                COMPLETE LIST OF QUANTUM DESIGN PATTERNS")
            print("=" * 80 + "\n")

            # 3. Print the results to the console
            for i, pattern in enumerate(detailed_patterns, 1):
                print(f"--- {i}. {pattern['name']} ---")
                for key, value in pattern.items():
                    if key != "name":
                        wrapped_text = textwrap.fill(
                            value,
                            width=100,
                            initial_indent="    ",
                            subsequent_indent="    ",
                        )
                        print(f"  {key.capitalize()}:\n{wrapped_text}")
                print("\n")

            # 4. Save the results to a file
            try:
                with open(QUANTUM_PATTERNS_REFERENCE_FILE, "w", encoding="utf-8") as f:
                    json.dump(detailed_patterns, f, indent=2, ensure_ascii=False)
                print("=" * 80)
                print(
                    f"Successfully saved {len(detailed_patterns)}/{len(patterns_to_process)} patterns to 'quantum_patterns.json'"
                )
                print("=" * 80)
            except OSError as e:
                print(f"Error: Could not save data to file. {e}")
