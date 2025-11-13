#!/usr/bin/env python3
"""
Interactive setup script for Mongolian History RAG
Helps configure API key and test the system
"""

import os
import sys
from pathlib import Path
from getpass import getpass


def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("🇲🇳 Mongolian History RAG - Setup")
    print("=" * 60)
    print()


def check_env_file():
    """Check if .env file exists."""
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        return True
    else:
        print("❌ .env file not found")
        return False


def create_env_file():
    """Create .env file from template."""
    print("\n📝 Creating .env file...")
    
    # Check if example exists
    example_file = Path('.env.example')
    if not example_file.exists():
        print("❌ .env.example not found")
        return False
    
    # Get API key
    print("\n🔑 OpenAI API Key Setup")
    print("-" * 60)
    print("Options:")
    print("1. Enter API key now")
    print("2. Skip (use Simple RAG without API key)")
    print("3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        api_key = getpass("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return False
        
        if not api_key.startswith('sk-'):
            print("⚠️  Warning: API key should start with 'sk-'")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False
        
        # Create .env file
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write("OPENAI_MODEL=gpt-4o-mini\n")
            f.write("OPENAI_TEMPERATURE=0.7\n")
        
        print("✅ .env file created")
        return True
    
    elif choice == "2":
        print("\n✅ Skipping API key setup")
        print("You can use Simple RAG without an API key")
        return False
    
    else:
        print("\n👋 Setup cancelled")
        sys.exit(0)


def test_api_key():
    """Test if API key works."""
    print("\n🧪 Testing API key...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found in environment")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        print("✅ API key is valid!")
        return True
    
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False


def show_next_steps(has_api_key):
    """Show what to do next."""
    print("\n" + "=" * 60)
    print("🎯 Next Steps")
    print("=" * 60)
    
    if has_api_key:
        print("\n✅ You can use all features:")
        print("\n1. Web Interface:")
        print("   python preview_ui.py")
        print("   Then open: http://localhost:5000")
        
        print("\n2. Command Line:")
        print("   python quick_rag_demo.py")
        
        print("\n3. Full RAG System:")
        print("   python rag_with_gpt.py")
    
    else:
        print("\n✅ You can use Simple RAG (no API key needed):")
        print("\n1. Simple RAG:")
        print("   python demo_simple_rag.py")
        
        print("\n2. UI Preview (static):")
        print("   open ui_preview.html")
        
        print("\n💡 To use GPT features later:")
        print("   1. Get API key from: https://platform.openai.com/api-keys")
        print("   2. Run: python setup.py")
    
    print("\n" + "=" * 60)


def main():
    """Main setup flow."""
    print_header()
    
    # Check if .env exists
    has_env = check_env_file()
    
    if has_env:
        # Load .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed")
            print("   Install: pip install python-dotenv")
        
        # Test API key
        has_api_key = test_api_key()
    
    else:
        # Create .env
        print("\n📋 .env file not found. Let's create one!")
        has_api_key = create_env_file()
        
        if has_api_key:
            # Load and test
            try:
                from dotenv import load_dotenv
                load_dotenv()
                has_api_key = test_api_key()
            except ImportError:
                print("⚠️  python-dotenv not installed")
                has_api_key = False
    
    # Show next steps
    show_next_steps(has_api_key)
    
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
