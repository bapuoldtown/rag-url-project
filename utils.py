import os
import yaml


class FileLoaded:
    @staticmethod
    def load_yaml_file():
        script_dir = "api_keys"
        file_path = os.path.join(script_dir, "apikeys.yml")
        
        #print(f"File path: {file_path}")
        #print(f"File exists: {os.path.exists(file_path)}")
        
        with open(file_path, 'r') as yaml_file:
            # Read the raw content first
            #yaml_file.seek(0)
            #raw_content = yaml_file.read()
            #print(f"Raw file content:\n{repr(raw_content)}")
            
            # Now parse it
            #yaml_file.seek(0)
            config = yaml.safe_load(yaml_file)
            
            #print(f"\nType of config: {type(config)}")
            #print(f"Config: {config}")
            
            if isinstance(config, dict):
                #print(f"Its a dictionary")
                #api_key = config['openai']['api_key']
                return config.get('openai').split(":")[-1]
                #return api_key
            else:
                print("ERROR: Config is not a dictionary!")
                return None


class ApiKeyHandler:
    @staticmethod
    def get_apikey():
        return FileLoaded.load_yaml_file()

if __name__ == "__main__":
    print("\nAPI_KEY:", ApiKeyHandler.get_apikey())