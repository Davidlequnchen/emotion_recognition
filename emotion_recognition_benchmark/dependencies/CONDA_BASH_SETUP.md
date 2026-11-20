# Install Miniconda on Linux x86_64

1. From the repository root, enter the benchmark folder and join the installer parts.

   ```bash
   cd emotion_recognition_benchmark
   python3 scripts/common/join_parts.py --file dependencies/Miniconda3-py311_26.9.4-1-Linux-x86_64.sh
   ```

2. Once you see `SHA-256 checked`, run the installer.

   ```bash
   bash dependencies/Miniconda3-py311_26.9.4-1-Linux-x86_64.sh -b -p "$HOME/miniconda3"
   ```

3. Initialize Conda for Bash.

   ```bash
   "$HOME/miniconda3/bin/conda" init bash
   ```

4. Open `~/.bash_profile` with `nano ~/.bash_profile`. Add this line once and save.

   ```bash
   [ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
   ```

5. Open a new terminal and run `conda --version`.
