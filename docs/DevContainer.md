## Develop Container
If you don't use bare meachine often, it's a good choice to develop in cloud (such as codespace, lightning.ai studio) with the help of [devcontainer](https://containers.dev/). To Test your `json` configuration, developer could use [devcontainer cli](https://github.com/devcontainers/cli).
```bash
# Install devcontaienr cli with npm
npm install -g @devcontainers/cli
```

```bash
# build and debug your json configuration
devcontainer build --workspace-folder . --log-level trace
```

```bash 
# Start up container first
devcontainer up --workspace-folder .

# entry bash inside container
devcontainer exec --workspace-folder . bash
```

### example
Here is an example based on GitHub default image in codespace
```json
{
  "image": "mcr.microsoft.com/devcontainers/universal:2",
  "features": {
    "ghcr.io/va-h/devcontainers-features/uv:1": {}
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python"
      ]
    }
  }
}
```
`universal` provide a basic development environment in alpine linux, and features named `uv`. `customizations` phase for visual studio code configuration, which often a personal prefence.