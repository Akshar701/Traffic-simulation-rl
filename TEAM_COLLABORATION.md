# Team Collaboration Guide

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-private-repo-url>
   cd traffic-simulation-rl
   ```

2. **Create your feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes and test them**

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   git push origin feature/your-feature-name
   ```

## 📋 Branch Naming Convention

- `feature/` - New features (e.g., `feature/dashboard-improvements`)
- `fix/` - Bug fixes (e.g., `fix/traffic-generation-bug`)
- `docs/` - Documentation updates (e.g., `docs/readme-update`)

## 🔄 Workflow

1. **Always pull latest changes** before starting work
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/your-branch
   git merge main
   ```

2. **Test your changes** before pushing

3. **Keep commits small and focused**

4. **Use descriptive commit messages**

## 🚨 Important Notes

- **Don't push directly to main** - use feature branches
- **Test your code** before committing
- **Update documentation** if you add new features
- **Ask for help** if you're stuck

## 🆘 Need Help?

- Check existing issues
- Ask in your team chat
- Review the code together
