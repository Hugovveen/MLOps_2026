# Assignment 2: MLOps & PCAM Pipeline Journal
**MLOps & ML Programming (2026)**

## Group Information
* **Group Number:** [Group 2]
* **Team Members:** [Eva van Wiggen/15246280, Hugo van Veen/15759261, Arjan van Staveren/13607812, David van Goethem/15628124, Joshua Appiah/13606069]
* **GitHub Repository:** [[Link to your Group Repository](https://github.com/Hugovveen/MLOps_2026)]
* **Base Setup Chosen from Assignment 1:** [Hugo van Veen]

---

## Question 1: Reproducibility Audit
1. **Sources of Non-Determinism:**


#### TRAIN/VAL Set
- Training and Validation shuffling causes the distribution of data points to be random, reducing the reproducibility of the run. (This happens in loader.py)


#### Dataloader config
- The configuration DataLoader(shuffle=True), creates a random permutation of indices for every epoch (iteration), this changes the sequence of gradients the model gets every run, breaking determinism and reproducibility. 

#### Hardware FPO's
- different sets of hardware could have different kind of rounding errors due to finite preciscion that floating point operations have (FPO's)

#### Libraries
- Different versions of libraries could possibly influence the behavior of certain operations, PyTorch, CUDA, Mod-stacks methods chang from version to version. This once again break determinism.



2. **Control Measures:**
- Use a random seed for the train/val set

- It is possible Turn shuffle off in the data loader, but this can hurt performance, you can also keep shuffling but via random seeds and deterministic settings.

- This is hard to fix but you can mitigate this issue by using deterministic settings in pytorch and CUDA.  

- Make sure the required libraries with versions are documented in a file for the exact run. like a requirements.txt file, also using commit messages in git helps!


3. **Code Snippets for Reproducibility:**
   ```python
   # Paste the exact code added for seeding and determinism
   ```

4. **Twin Run Results:**

---

## Question 2: Data, Partitioning, and Leakage Audit
1. **Partitioning Strategy:**

2. **Leakage Prevention:**
   
3. **Cross-Validation Reflection:**

4. **The Dataset Size Mystery:**

5. **Poisoning Analysis:**

---

## Question 3: Configuration Management
1. **Centralized Parameters:**

2. **Loading Mechanism:**
   - [Describe your use of YAML, Hydra, or Argparse.]
   ```python
   # Snippet showing how parameters are loaded
   ```

3. **Impact Analysis:**

4. **Remaining Risks:** 

---

## Question 4: Gradients & LR Scheduler
1. **Internal Dynamics:**

2. **Learning Rate Scheduling:**

---

## Question 5: Part 1 - Experiment Tracking
1. **Metrics Choice:**

2. **Results (Average of 3 Seeds):**

3. **Logging Scalability:**

4. **Tracker Initialization:**
   ```python
   # Snippet showing tracker/MLFlow/W&B initialization
   ```

5. **Evidence of Logging:**

6. **Reproduction & Checkpoint Usage:**

7. **Deployment Issues:**

---

## Question 5: Part 2 - Hyperparameter Optimization
1. **Search Space:**
2. **Visualization:**
3. **The "Champion" Model:**

4. **Thresholding Logic:**

5. **Baseline Comparison:**

---

## Question 6: Model Slicing & Error Analysis
1. **Visual Error Patterns:**

2. **The "Slice":**

3. **Risks of Silent Failure:**

---

## Question 7: Team Collaboration and CI/CD
1. **Consolidation Strategy:** 
2. **Collaborative Flow:**

3. **CI Audit:**

4. **Merge Conflict Resolution:**

5. **Branching Discipline:**

---

## Question 8: Benchmarking Infrastructure
1. **Throughput Logic:**

2. **Throughput Table (Batch Size 1):**

| Partition | Node Type | Throughput (img/s) | Job ID |
| :--- | :--- | :--- | :--- |
| `thin_course` | CPU Only | | |
| `gpu_course` | GPU ([Type]) | | |

3. **Scaling Analysis:**

4. **Bottleneck Identification:**

---

## Question 9: Documentation & README
1. **README Link:** [Link to your Group Repo README]
2. **README Sections:** [Confirm Installation, Data Setup, Training, and Inference are present.]
3. **Offline Handover:** [List the files required on a USB stick to run the model offline.]

---

## Final Submission Checklist
- [ ] Group repository link provided?
- [ ] Best model checkpoint pushed to GitHub?
- [ ] inference.py script included and functional?
- [ ] All Slurm scripts included in the repository?
- [ ] All images use relative paths (assets/)?
- [ ] Names and IDs of all members on the first page?