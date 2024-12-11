import os
from pathlib import Path

def load_api_key():
    """Load OpenRouter API key from environment variable or prompt user."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        # Check for API key in .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('OPENROUTER_API_KEY='):
                        api_key = line.split('=')[1].strip()
                        break
    
    # Prompt user if still no API key
    if not api_key:
        api_key = input("Please enter your OpenRouter API key: ").strip()
        save = input("Would you like to save this API key to a .env file? (y/N): ").lower()
        
        if save == 'y':
            with open('.env', 'a') as f:
                f.write(f"\nOPENROUTER_API_KEY={api_key}")
            print("API key saved to .env file")
    
    return api_key