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

```bashrc
GEMINI_API_KEY=
```

Iy you face rate limit error, you can 5 more api key from multiple accounts

```env
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
```

Recommeded way is use 6 gemini api from 6 accounts for 6 games to play concurrently.

Also,
3 api key is recommeded to play 3 games concurrently.
1 api key will be enough, if you have higher gemini pla

5. Run the playzero agent (generates random actions) against all the games.

```bash
uv run main.py --agent=playzeroagent
```

**Note: for linux users**

We are using opencv, this might lead this error when you run

```bash
    native_module = importlib.import_module("cv2")
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/codespace/.python/current/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

OpenCV (cv2) is trying to use OpenGL for image display or processing, but if the shared library libGL.so.1 isn’t installed in your environment.
This is common in minimal Docker/virtual environments (like Codespaces).

How to fix it:
You just need to install the missing system dependency.

```shell
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```


For more information, see the [documentation](https://three.arcprize.org/docs#quick-start) or the [tutorial video](https://youtu.be/xEVg9dcJMkw).

## Architecture

### Play Zero Agent

**Steps**
1. Random play (do random probability after a threshold of actions met)
2. Do analysis and create goal with max actions limit(1 Video LLM call, 1 text LLM call)
3. Takes actions using (1 Image LLM will current frame), goal achievent check (2 Images LLM call). this step is repeated based on 4th and 5th step.
4. If max actions limit reached without achieving or level changed, do exploring (with random play) 
5. If goal achieved, do game analysis
6. Iterated till max actions. current is 2500 actions.

**Challenges**
- The Video LLM call will take time.
- We can switch to gemini-2.5-pro, it will give good performance. But there might rate limit errors.
- There would be about 250K tokens needed to clear level 1. THis imght increase by 33% for each level. About 250 actions might be needed to clear level 1. this will be increaseing by 33% as well.
- So, the token calculation for 5 levels and actions calculations

Using gemini API for LLM

Here’s the clean summary:

| Level     | Tokens           | Actions      |
| --------- | ---------------- | ------------ |
| 1         | 250,000          | 250          |
| 2         | 332,500          | 332.5        |
| 3         | 442,225          | 442.225      |
| 4         | 587,154.25       | 587.15425    |
| 5         | 780,910.1525     | 780.91015    |
| **Total** | **2,392,789.40** | **2,392.79** |


| Level     | Tokens           | Flash Cost (\$) | Pro Cost (\$) |
| --------- | ---------------- | --------------- | ------------- |
| 1         | 250,000          | 0.6250          | 3.7500        |
| 2         | 332,500          | 0.83125         | 4.9875        |
| 3         | 442,225          | 1.1055625       | 6.633375      |
| 4         | 587,154.25       | 1.4678856       | 8.8073138     |
| 5         | 780,910.1525     | 1.9522754       | 11.713652     |
| **Total** | **2,392,789.40** | **5.9820**      | **35.8918**   |


The Gemini API cost might range from $5 to $35 dollors

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
