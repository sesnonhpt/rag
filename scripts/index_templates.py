#!/usr/bin/env python3
"""Script to index all template files for metadata search."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv
load_dotenv()

from app.services.template_metadata_service import TemplateMetadataService
from app.core.paths import APP_ROOT

def main():
    """Index all template files."""
    templates_dir = APP_ROOT.parent / "data" / "templates"
    
    print(f"Indexing templates in: {templates_dir}")
    print("-" * 60)
    
    # Create service (will try LLM if available, fallback to rule-based)
    service = TemplateMetadataService(templates_dir, use_llm=True)
    
    # Index all files (force=True to re-index everything)
    service.index_all(force=True)
    
    print("-" * 60)
    print(f"✓ Indexing complete!")
    print(f"  Total indexed: {len(service._index)}")
    print(f"  Metadata file: {service.metadata_file}")
    print(f"  LLM enabled: {service.use_llm}")
    
    # Show sample results
    print("\nSample indexed files:")
    for i, (filename, metadata) in enumerate(list(service._index.items())[:5]):
        print(f"\n{i+1}. {filename}")
        print(f"   Desc: {metadata.desc[:80]}...")
        print(f"   Keywords: {', '.join(metadata.keywords[:5])}")

if __name__ == "__main__":
    main()
