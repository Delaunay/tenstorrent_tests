




venv:
	uv venv --python=3.12
	(. .venv/bin/activate && uv pip install -r requirements.txt)

run: venv
	(. .venv/bin/activate && DEVICE="tt" python script.py > log_tt.stdout 2> log_tt.stderr)
	(. .venv/bin/activate && DEVICE="cpu" python script.py > log_cpu.stdout 2> log_cpu.stderr)