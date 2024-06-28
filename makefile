# Makefile

# load envs 
include .env
export $(shell sed 's/=.*//' .env)


.PHONY: print-env
print-env:
	@echo "LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2}"
	@echo "LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}"
	@echo "OPENAI_API_KEY=${OPENAI_API_KEY}"


.PHONY: req
req:
	@uv pip compile requirements.in -o requirements.txt
	@uv pip sync requirements.txt 

.PHONY: chroma-example
chroma-example:
	@python chroma-example.py

.PHONY: rag-example
rag-example:
	@python rag-example.py