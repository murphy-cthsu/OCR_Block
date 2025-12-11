import base64
import json
import re
from openai import OpenAI
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import os
from dotenv import load_dotenv


def clean_json_string(json_str: str) -> str:
    """Clean common JSON issues from LLM responses."""
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # Remove comments (// style)
    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
    # Remove comments (/* */ style)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    # Fix unquoted property names (simple cases)
    json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    return json_str

class BlockDiagramAnnotator:
    def __init__(self, api_key: str = None):
        # Load from .env file if it exists
        load_dotenv()
        
        # Use provided api_key, fall back to environment variable
        final_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not final_api_key:
            raise ValueError(
                "API key not provided. Please either:\n"
                "  1. Pass api_key parameter to BlockDiagramAnnotator\n"
                "  2. Set OPENROUTER_API_KEY environment variable\n"
                "  3. Create a .env file with OPENROUTER_API_KEY=your-key"
            )
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=final_api_key
        )
        self.primary_model = "anthropic/claude-sonnet-4.5"
        self.secondary_model = "openai/gpt-5.1"
        self.tertiary_model = "mistral/mistral-7b-instruct-v0.1"

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_annotation_prompt(self, image_b64: str, instructions: str, pass_number: int = 1) -> str:
        """Create annotation prompt for block diagram analysis.
        
        Args:
            image_b64: Base64 encoded image
            instructions: User instructions
            pass_number: Analysis pass number (1, 2, or 3)
        
        Returns:
            Formatted prompt string
        """
        base_prompt = """You are an expert hardware engineer analyzing a block diagram for RTL implementation.

Your task is to extract ALL information with EXTREME precision. This will be used to generate RTL code.

Analyze this block diagram and provide a complete, accurate JSON output with the following structure:

{
  "modules": [
    {
      "id": "unique_module_id",
      "name": "module_name",
      "type": "module_type (e.g., register, mux, alu, fifo, memory, custom)",
      "position": {"x": approximate_x, "y": approximate_y},
      "parameters": {
        "width": "bit_width_if_specified",
        "depth": "depth_if_memory",
        "other_params": "any other parameters shown"
      },
      "description": "any additional notes about this module"
    }
  ],
  "connections": [
    {
      "id": "unique_connection_id",
      "from_module": "source_module_id",
      "from_port": "output_port_name",
      "to_module": "destination_module_id",
      "to_port": "input_port_name",
      "signal_name": "signal_name_if_labeled",
      "width": "bit_width",
      "is_bus": true,
      "direction": "arrow direction",
      "annotations": ["any labels or notes on the connection"]
    }
  ],
  "ports": [
    {
      "name": "port_name",
      "direction": "input/output/inout",
      "width": "bit_width",
      "connected_to": "module_id",
      "description": "port description if available"
    }
  ],
  "annotations": [
    {
      "type": "text/label/note",
      "content": "annotation content",
      "location": "Description of where the annotation is located on the diagram",
      "related_to": "module_id/connection_id if applicable"
    }
  ],
  "hierarchy": {
    "top_level_module": "name if specified",
    "submodules": ["list of identified submodules"],
    "interfaces": ["identified interfaces like AXI, AMB, etc."]
  },
  "metadata": {
    "total_modules": 0,
    "total_connections": 0,
    "complexity": "simple/moderate/complex",
    "protocol_hints": ["any protocol standards detected"],
    "ambiguities": ["list any unclear or ambiguous elements"]
  }
}

CRITICAL REQUIREMENTS:
1. Extract EVERY module, even small ones
2. Identify ALL connections with exact signal names
3. Capture bit widths wherever shown (e.g., [7:0], [31:0])
4. Note direction of ALL arrows
5. Include ALL text labels and annotations
6. If you see "/" or "-" notation, that indicates bit ranges
7. If multiple signals go to same destination, list each separately
8. Pay attention to bus notation (thick lines, multiple arrows)
9. Identify any clock, reset, or control signals explicitly
10. Note any feedback loops or bidirectional connections

VALIDATION:
- Every connection must have valid from_module and to_module
- Module IDs must be consistent across modules and connections
- If unsure about something, note it in ambiguities
"""
        
        if pass_number == 2:
            base_prompt += """
SECOND PASS INSTRUCTIONS:
You are reviewing a previous analysis. Focus on:
1. Catching missed connections (especially small or faint ones)
2. Verifying bit widths are correct
3. Finding any unlabeled signals
4. Checking for implicit connections (power, ground, clock distribution)
5. Validating signal name consistency
"""
        elif pass_number == 3:
            base_prompt += """
THIRD PASS INSTRUCTIONS:
Final verification pass. Focus on:
1. Cross-checking connection consistency
2. Ensuring no duplicate IDs
3. Ensuring all ports are accounted for
4. Checking for any overlooked text annotations
5. Validating the overall architecture makes sense
"""
        
        return base_prompt
    
    def annotate_with_model(self, image_path:str, model:str, pass_number:int =1 ) -> Dict:
        image_b64 = self.encode_image(image_path)
        image_ext=Path(image_path).suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }.get(image_ext, 'image/png')
        prompt = self.create_annotation_prompt(image_b64, instructions="", pass_number=pass_number)
        
        try :
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content":[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=16000,
            )
            content = response.choices[0].message.content
            
            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"  Warning: Response truncated due to max_tokens limit")
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Clean common JSON issues from LLM responses
            content = clean_json_string(content)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as je:
                print(f"  JSON parse error at position {je.pos}: {je.msg}")
                print(f"  Problematic content around error: ...{content[max(0, je.pos-50):je.pos+50]}...")
                return None
    
        except Exception as e:
            error_msg = str(e)
            # Check for fatal API errors that should stop the entire process
            if "Key limit exceeded" in error_msg or "Insufficient credits" in error_msg:
                raise e
                
            print(f"Error with model {model} on pass {pass_number}: {e}")
            return None
        
    def merge_annotations(
        self,
        annotations: List[Dict]
    ) -> Dict:
        """
        Merge multiple annotation passes using consensus and union
        Priority: Information that appears in multiple passes is most reliable
        """

        if not annotations:
            return {}

        # Filter out None values
        valid_annotations = [a for a in annotations if a]

        merged = {
            "modules": [],
            "connections": [],
            "ports": [],
            "annotations": [],
            "hierarchy": {},
            "metadata": {
                "merge_strategy": "consensus_with_union",
                "sources_count": len(valid_annotations)
            }
        }

        # Merge modules - use union with confidence scoring
        module_map = {}   # name -> list of module definitions

        for ann in valid_annotations:
            for module in ann.get("modules", []):
                name = module.get("name", "")
                # Use lowercase name for better grouping
                key = name.lower().strip() if name else ""
                if key not in module_map:
                    module_map[key] = []
                module_map[key].append(module)

        for key, module_instances in module_map.items():
            # Use most detailed version (most keys)
            best_module = max(module_instances, key=lambda m: len(json.dumps(m)))
            best_module["confidence"] = len(module_instances) / len(valid_annotations)
            merged["modules"].append(best_module)

        # Merge connections
        connection_map = {}  # (from, to) -> list of connections

        for ann in valid_annotations:
            for conn in ann.get("connections", []):
                # Use lowercase IDs for grouping
                from_mod = conn.get("from_module", "").lower().strip()
                to_mod = conn.get("to_module", "").lower().strip()
                key = (from_mod, to_mod)
                
                if key not in connection_map:
                    connection_map[key] = []
                connection_map[key].append(conn)

        for key, conn_instances in connection_map.items():
            best_conn = max(conn_instances, key=lambda c: len(json.dumps(c)))
            best_conn["confidence"] = len(conn_instances) / len(valid_annotations)
            merged["connections"].append(best_conn)

        # Merge ports
        port_map = {}

        for ann in valid_annotations:
            for port in ann.get("ports", []):
                name = port.get("name", "")
                if name not in port_map:
                    port_map[name] = []
                port_map[name].append(port)

        for name, port_instances in port_map.items():
            best_port = max(port_instances, key=lambda p: len(json.dumps(p)))
            best_port["confidence"] = len(port_instances) / len(valid_annotations)
            merged["ports"].append(best_port)

        # Merge annotations (text labels / notes)
        all_annotations = []
        for ann in valid_annotations:
            all_annotations.extend(ann.get("annotations", []))

        # Deduplicate based on content similarity
        unique_annotations = []
        seen_content = set()

        for annotation in all_annotations:
            content = annotation.get("content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                unique_annotations.append(annotation)

        merged["annotations"] = unique_annotations

        # Take hierarchy from most detailed source
        for ann in valid_annotations:
            if ann.get("hierarchy") and len(ann["hierarchy"]) > len(merged["hierarchy"]):
                merged["hierarchy"] = ann["hierarchy"]

        # Aggregate metadata
        merged["metadata"]["total_modules"] = len(merged["modules"])
        merged["metadata"]["total_connections"] = len(merged["connections"])

        # Collect all ambiguities
        all_ambiguities = []
        for ann in valid_annotations:
            all_ambiguities.extend(ann.get("metadata", {}).get("ambiguities", []))

        merged["metadata"]["ambiguities"] = list(set(all_ambiguities))

        return merged

    def post_process_annotation(self, annotation: Dict) -> Dict:
        """
        Clean up and normalize annotation data.
        - Normalizes IDs to lowercase
        - Deduplicates modules with same ID
        - Ensures connection references match module IDs
        """
        if not annotation:
            return annotation

        def norm(s):
            return str(s).lower().strip().replace(" ", "_") if s else ""

        # 1. Normalize Module IDs and Deduplicate
        unique_modules = {}
        for module in annotation.get("modules", []):
            mid = norm(module.get("id", ""))
            if not mid: continue
            
            module["id"] = mid
            
            if mid not in unique_modules:
                unique_modules[mid] = module
            else:
                # If duplicate ID found, merge/keep the one with more details
                existing = unique_modules[mid]
                if len(json.dumps(module)) > len(json.dumps(existing)):
                    unique_modules[mid] = module
        
        annotation["modules"] = list(unique_modules.values())
        
        # 2. Normalize Connections
        for conn in annotation.get("connections", []):
            if "from_module" in conn:
                conn["from_module"] = norm(conn["from_module"])
            if "to_module" in conn:
                conn["to_module"] = norm(conn["to_module"])
            
        # 3. Normalize Ports
        for port in annotation.get("ports", []):
            if "connected_to" in port:
                port["connected_to"] = norm(port["connected_to"])

        # 4. Normalize Annotations
        for note in annotation.get("annotations", []):
            if "related_to" in note:
                note["related_to"] = norm(note["related_to"])
                
        return annotation

    def validate_annotation(self, annotation: Dict) -> Dict:
        """
        Validate annotation for consistency and completeness
        Returns validation report
        """

        issues = []
        warnings = []

        modules = annotation.get("modules", [])
        connections = annotation.get("connections", [])

        # Check for duplicate module IDs
        module_ids = [m.get("id", "") for m in modules]
        if len(module_ids) != len(set(module_ids)):
            issues.append("Duplicate module IDs detected")

        # Check connection validity
        module_id_set = set(module_ids)
        for i, conn in enumerate(connections):
            from_mod = conn.get("from_module", "")
            to_mod = conn.get("to_module", "")

            if from_mod not in module_id_set:
                issues.append(f"Connection {i}: from_module {from_mod} not found")

            if to_mod not in module_id_set:
                issues.append(f"Connection {i}: to_module {to_mod} not found")

        # Check for isolated modules (no connections)
        connected_modules = set()
        for conn in connections:
            connected_modules.add(conn.get("from_module", ""))
            connected_modules.add(conn.get("to_module", ""))

        isolated = module_id_set - connected_modules
        if isolated:
            warnings.append(f"Isolated modules (no connections): {isolated}")

        # Check bit width consistency
        for conn in connections:
            if conn.get("width") == 0:
                warnings.append(
                    f"Connection {conn.get('signal_name')} has width 0"
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": {
                "modules": len(modules),
                "connections": len(connections),
                "ports": len(annotation.get("ports", []))
            }
        }

    def annotate(
        self,
        image_path: str,
        multi_pass: bool = True,
        multi_model: bool = False,
        save_intermediate: bool = False,
        output_dir: str = None,
        output_prefix: str = None
    ) -> Dict:
        """
        Main annotation function with multiple passes and models for accuracy

        Args:
            image_path: Path to block diagram image
            multi_pass: Whether to do multiple passes with same model
            multi_model: Whether to use multiple models for cross-validation
            save_intermediate: Save intermediate results for debugging
            output_dir: Directory to save intermediate files (default: current dir)
            output_prefix: Prefix for output files (default: derived from image name)
        """

        print(f"Starting high-accuracy annotation of: {image_path}")
        all_annotations = []

        # Set up output paths
        if output_prefix is None:
            output_prefix = Path(image_path).stem
        if output_dir is None:
            output_dir = "."
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Strategy for maximum accuracy:
        # 1. Multiple passes with primary model (catch progressive details)
        # 2. Cross-validation with other models
        # 3. Merge with consensus

        models_to_use = [self.primary_model]
        if multi_model:
            models_to_use.extend([self.secondary_model, self.tertiary_model])

        for model_idx, model in enumerate(models_to_use):
            print("\n" + "="*60)
            print(f"Processing with model: {model}")
            print("="*60)

            passes = 3 if (multi_pass and model_idx == 0) else 1

            for pass_num in range(1, passes + 1):
                print(f"\nPass {pass_num}/{passes}...")

                annotation = self.annotate_with_model(
                    image_path,
                    model,
                    pass_number=pass_num
                )

                if annotation:
                    all_annotations.append(annotation)

                if save_intermediate:
                    model_safe = model.replace('/', '_')
                    output_file = output_dir / f"{output_prefix}_intermediate_{model_safe}_pass{pass_num}.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(annotation, f, indent=2)
                    print(f"Saved intermediate result to {output_file}")

            if annotation:
                print(f" Connections found: {len(annotation.get('connections', []))}")
                print(f" Modules found: {len(annotation.get('modules', []))}")

            # Rate limiting / cooling period between calls
            if pass_num < passes or model_idx < len(models_to_use) - 1:
                time.sleep(2)

        print("\n" + "="*60)
        print(f"Merging {len(all_annotations)} annotation results...")
        print("="*60)

        # Merge all annotations
        final_annotation = self.merge_annotations(all_annotations)

        # Post-process to regularize IDs and fix case mismatches
        final_annotation = self.post_process_annotation(final_annotation)

        # Validate
        validation = self.validate_annotation(final_annotation)
        final_annotation["validation"] = validation

        print("\nValidation Results:")
        print(f"  ✓ Valid: {validation['valid']}")
        print(f"  ✗ Issues: {len(validation['issues'])}")
        print(f"  ⚠ Warnings: {len(validation['warnings'])}")

        if validation["issues"]:
            print("\nIssues found:")
            for issue in validation["issues"]:
                print(f"  - {issue}")

        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"][:5]:  # show first 5
                print(f"  - {warning}")

        print("\nFinal Statistics:")
        print(f"  Total modules: {validation['stats']['modules']}")
        print(f"  Total connections: {validation['stats']['connections']}")
        print(f"  Total ports: {validation['stats']['ports']}")

        return final_annotation

def main():
    annotator = BlockDiagramAnnotator()
    image_path = "images/MT6797.png"
    result = annotator.annotate(
        image_path,
        multi_pass=True,
        multi_model=True,
        save_intermediate=True
    )
    with open("final_annotation.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Final annotation saved to final_annotation.json")

    # Optional: Pretty print a summary
    print("\n" + "="*60)
    print("ANNOTATION SUMMARY")
    print("="*60)

    print(f"\nModules ({len(result['modules'])}): show first 5")
    for mod in result["modules"][:5]:
        conf = mod.get("confidence", 1.0)
        print(f"  - {mod['name']} ({mod['type']}) [confidence: {conf:.2f}]")

    print(f"\nConnections ({len(result['connections'])}): show first 5")
    for conn in result["connections"][:5]:
        signal = conn.get("signal_name", "unnamed")
        width = conn.get("width", "?")
        conf = conn.get("confidence", 1.0)
        print(
            f"  - {conn['from_module']} -> {conn['to_module']}: "
            f"{signal} [{width}] [confidence: {conf:.2f}]"
        )
        width = conn.get("width", "?")
        conf = conn.get("confidence", 1.0)
        print(
            f"  - {conn['from_module']} -> {conn['to_module']}: "
            f"{signal} [{width}] [confidence: {conf:.2f}]"
        )

if __name__ == "__main__":
    main()