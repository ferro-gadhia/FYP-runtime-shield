from sandbox_env import SandboxEnv


env = SandboxEnv()

env.execute({"type": "read_file", "path": "/sandbox/report.txt"})
env.execute({"type": "read_file", "path": "/secret/passwords.txt"})

env.execute({"type": "write_file", "path": "/sandbox/new.txt", "content": "hello"})
env.execute({"type": "write_file", "path": "/exfil/out.txt", "content": "leak"})
env.execute({"type": "write_file", "path": "/secret/leak.txt", "content": "oops"})

env.execute({"type": "scan_network", "target_range": "10.0.0.0/24"})
env.execute({"type": "scan_network", "target_range": "192.168.0.0/16"})

for h in env.history:
    print(h)
