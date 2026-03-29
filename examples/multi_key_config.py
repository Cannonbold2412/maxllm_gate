"""Example: Configuring multiple API keys."""

import json

# Example configuration for multiple API keys across providers
# Save this to .env or set as API_KEYS_CONFIG environment variable

config = {
    # Groq keys (fast, free tier with limits)
    "groq-key-1": {
        "api_key": "gsk_your_groq_api_key_1",
        "provider": "groq",
        "models": [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "tpm_limit": 30000,  # Tokens per minute
        "rpm_limit": 30,     # Requests per minute
    },
    "groq-key-2": {
        "api_key": "gsk_your_groq_api_key_2",
        "provider": "groq",
        "models": [
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        "tpm_limit": 30000,
        "rpm_limit": 30,
    },
    
    # OpenRouter keys (access to many models)
    "openrouter-1": {
        "api_key": "sk-or-your_openrouter_key",
        "provider": "openrouter",
        "models": [
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet",
            "meta-llama/llama-3-70b-instruct",
            "mistralai/mixtral-8x7b-instruct",
        ],
        "tpm_limit": 100000,
        "rpm_limit": 200,
    },
    
    # OpenAI keys
    "openai-primary": {
        "api_key": "sk-your_openai_key",
        "provider": "openai",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
        "tpm_limit": 90000,
        "rpm_limit": 500,
    },
    "openai-backup": {
        "api_key": "sk-your_backup_openai_key",
        "provider": "openai",
        "models": [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        "tpm_limit": 60000,
        "rpm_limit": 300,
    },
    
    # NVIDIA NIM (self-hosted or cloud)
    "nvidia-nim-1": {
        "api_key": "nvapi-your_nvidia_key",
        "provider": "nvidia_nim",
        "models": [
            "meta/llama3-70b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
        "tpm_limit": 100000,
        "rpm_limit": 100,
    },
}

# Print as environment variable format
print("# Add to your .env file:")
print(f"API_KEYS_CONFIG='{json.dumps(config)}'")

print("\n# Or export directly:")
print(f"export API_KEYS_CONFIG='{json.dumps(config)}'")

# Show capacity summary
print("\n# Capacity Summary:")
total_tpm = sum(k["tpm_limit"] for k in config.values())
total_rpm = sum(k["rpm_limit"] for k in config.values())
print(f"# Total TPM: {total_tpm:,}")
print(f"# Total RPM: {total_rpm:,}")
print(f"# Keys: {len(config)}")
print(f"# Providers: {len(set(k['provider'] for k in config.values()))}")
