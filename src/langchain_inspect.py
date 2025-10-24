import pkgutil

def list_modules(package_name):
    try:
        package = __import__(package_name)
        print(f"Modules in {package_name}:")
        for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
            print(f"  {'[pkg]' if ispkg else '[mod]'} {modname}")
    except Exception as e:
        print(f"Error importing {package_name}: {e}")

if __name__ == "__main__":
    for pkg in [
        "langchain",
        "langchain_community",
        "langchain_openai"
    ]:
        list_modules(pkg)
