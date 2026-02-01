import yaml
import dotenv
from pathlib import Path

config_dir = Path(__file__).parent.parent.resolve() / "config"

# Load YAML config
with open(config_dir / "config.yml", 'r', encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

# Load .env config
config_env = dotenv.dotenv_values(config_dir / "config.env")

# Debugging prints to verify environment variables
print("Loaded config_env:", config_env)  # Check if environment variables are loaded correctly

# Configuration parameters
telegram_token = config_yaml["telegram_token"]
openai_api_key = config_yaml["openai_api_key"]
openai_api_base = config_yaml.get("openai_api_base", None)
allowed_telegram_usernames = config_yaml["allowed_telegram_usernames"]
new_dialog_timeout = config_yaml["new_dialog_timeout"]
enable_message_streaming = config_yaml.get("enable_message_streaming", True)
return_n_generated_images = config_yaml.get("return_n_generated_images", 1)
image_size = config_yaml.get("image_size", "512x512")
n_chat_modes_per_page = config_yaml.get("n_chat_modes_per_page", 5)
daily_question_chat_id=-1003501761776  # Replace with actual chat ID
# Define the MongoDB URI after loading the environment variables
mongodb_uri = (
    f"mongodb://{config_env.get('MONGODB_USERNAME', 'root')}:"
    f"{config_env.get('MONGODB_PASSWORD', 'example')}@"
    f"mongo:{config_env['MONGODB_PORT']}/"
    f"?authSource=admin"
)
# Debugging print to verify mongodb_uri
print("Constructed mongodb_uri:", mongodb_uri)  # Ensure mongodb_uri is being created correctly

# chat_modes
with open(config_dir / "chat_modes.yml", 'r') as f:
    chat_modes = yaml.safe_load(f)

# Models
with open(config_dir / "models.yml", 'r', encoding="utf-8") as f:
    models = yaml.safe_load(f)

# Files
help_group_chat_video_path = Path(__file__).parent.parent.resolve() / "static" / "help_group.mp4"
