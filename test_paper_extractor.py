#!/usr/bin/env python3
"""
Test script for running the PaperTaskExtractor agent on research papers.
This script demonstrates how to use the env_collection agent to extract
computational tasks, databases, and software from biomedical papers.
"""

import os
import json
from pathlib import Path
from biomni.agent.env_collection import PaperTaskExtractor


def test_llm_connection(agent):
    """Test LLM connectivity with a simple question."""
    print("\n🤖 Testing LLM connection...")
    try:
        # Send a simple test message
        response = agent.llm.invoke("Hello! Please respond with 'Hello, I am Claude' and your model name.")
        print(f"✅ LLM Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ LLM Connection failed: {str(e)}")
        return False


def read_paper_file(file_path: str) -> str:
    """Read a paper file (supports .txt, .md files)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    """Main function to test the PaperTaskExtractor agent."""
    
    # Initialize the agent
    print("🔬 Initializing PaperTaskExtractor agent...")
    
    # You can customize these parameters:
    agent = PaperTaskExtractor(
        llm="claude-3-7-sonnet-20250219",  # Or use "gpt-4" if you prefer
        chunk_size=4000,                   # Size of text chunks
        chunk_overlap=400                  # Overlap between chunks
    )
    
    # Test LLM connection first
    if not test_llm_connection(agent):
        print("❌ Cannot proceed without LLM connection. Please check your API key and network.")
        return
    
    # Define paths for your papers
    paper_paths = [
        "converted_papers/nature2.txt"   # Your converted Nature paper 2
    ]
    
    # Create results directory
    results_dir = Path("paper_extraction_results")
    results_dir.mkdir(exist_ok=True)
    
    # Process each paper
    for i, paper_path in enumerate(paper_paths, 1):
        if not os.path.exists(paper_path):
            print(f"❌ Paper file not found: {paper_path}")
            print(f"   Please make sure the file exists or update the path.")
            continue
            
        print(f"\n📄 Processing Paper {i}: {paper_path}")
        print("-" * 50)
        
        try:
            # Read the paper text
            paper_text = read_paper_file(paper_path)
            print(f"✅ Read paper: {len(paper_text)} characters")
            
            # Process the paper
            print("🔍 Extracting tasks, databases, and software...")
            log, results = agent.go(paper_text)
            
            # Display summary results
            print(f"\n📊 Extraction Results for Paper {i}:")
            print(f"   📋 Tasks found: {len(results.get('tasks', []))}")
            print(f"   🗄️  Databases found: {len(results.get('databases', []))}")
            print(f"   ⚙️  Software found: {len(results.get('software', []))}")
            
            # Save results to file
            output_file = results_dir / f"paper_2_results.json"
            agent.save_results(results, str(output_file))
            
            # Display some sample results
            if results.get('tasks'):
                print(f"\n🔬 Sample tasks from Paper {i}:")
                for j, task in enumerate(results['tasks'][:3], 1):  # Show first 3 tasks
                    print(f"   {j}. {task.get('task_name', 'Unknown task')}")
                    
            if results.get('databases'):
                print(f"\n🗄️ Sample databases from Paper {i}:")
                for j, db in enumerate(results['databases'][:3], 1):  # Show first 3 databases
                    print(f"   {j}. {db.get('name', 'Unknown database')}")
                    
            if results.get('software'):
                print(f"\n⚙️ Sample software from Paper {i}:")
                for j, sw in enumerate(results['software'][:3], 1):  # Show first 3 software
                    print(f"   {j}. {sw.get('name', 'Unknown software')}")
            
        except Exception as e:
            print(f"❌ Error processing {paper_path}: {str(e)}")
            continue
    
    print(f"\n✅ Processing complete! Results saved in '{results_dir}' directory.")
    print(f"💡 You can examine the detailed JSON files for full extraction results.")


if __name__ == "__main__":
    main()