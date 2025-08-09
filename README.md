# ARC-AGI-3-Agents

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.

1. Clone the ARC-AGI-3-Agents repo and enter the directory.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Copy .env-example to .env

```bash
cp .env-example .env
```

3. Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable in your .env file.

```bash
export ARC_API_KEY="your_api_key_here"
```

4. Get a Gemini API key from [Google AI studio](https://aistudio.google.com/apikey) and set it as an environemnt varaible in you .env file

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Iy you face rate limit error, you can two more api key from multiple accounts

```
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
```

5. Run the playzero agent (generates random actions) against all the games.

```bash
uv run main.py --agent=playzeroagent
```

For more information, see the [documentation](https://three.arcprize.org/docs#quick-start) or the [tutorial video](https://youtu.be/xEVg9dcJMkw).



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
