# Read installed packages
with open('installed_packages.txt') as f:
    installed = set(f.readlines())

# Read poetry tracked packages
with open('poetry_packages.txt') as f:
    poetry_tracked = set(f.readlines())

# Find differences
extra_in_installed = installed - poetry_tracked
missing_from_installed = poetry_tracked - installed

# Output results
print("Packages installed but not tracked by Poetry:")
print(extra_in_installed)

print("\nPackages tracked by Poetry but not installed:")
print(missing_from_installed)
