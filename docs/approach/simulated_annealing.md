## Optimization Process

1. **Neighbor generation**  
   - Perturb current architecture by tweaking each hyperparameter via `modify_value`.

2. **Evaluation & Δ-score**  
   - Compute  
     \[
       \Delta = \text{score(new)} - \text{score(current)}
     \]

3. **Acceptance test**  
   - **If** Δ > 0 (improvement):  
     - Always accept.  
     - Update best if it beats the record.  
   - **Else** (worse):  
     - Accept with probability  
       \[
         p = \exp\bigl(-\Delta / T\bigr)
       \]

4. **Stagnation escape**  
   - If \(|\Delta|\) remains below the threshold for too many steps, jump to a random past state from the archive.

5. **Cooling**  
   - Update temperature \(T\) via the chosen schedule (exponential, linear, or logarithmic).

Repeat until the iteration limit is reached, then return the best configuration found.
