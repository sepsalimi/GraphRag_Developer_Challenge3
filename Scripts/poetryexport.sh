poetry check && (poetry export -h >/dev/null 2>&1 || poetry self add poetry-plugin-export) && poetry export --extras openai -f requirements.txt -o requirements.txt --without-hashes
