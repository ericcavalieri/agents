## Agents
Um simples sistema de agents

## Python
Version: 3.11

## Envs
Para o correto funcionamento, deverá copiar o .env.example para .env e ajustar os valores internos, sendo:
```python
COHERE_API_KEY=COHERE_APY_KEY
AGENT_TYPE=[agent_agenda | agent_email | multi_agent]
```

## Run
```
pip install -r requirements.txt
```

```
pf flow serve --source .
```