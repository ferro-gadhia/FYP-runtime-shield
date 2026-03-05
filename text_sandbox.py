from sandbox_env import SandboxEnv
env = SandboxEnv()
print("Initiat state:")
print(env.get_state_summary)

print("\nFilesystem")
for path in env.filesystem:
    print(" -", path)