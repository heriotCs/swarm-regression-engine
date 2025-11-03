import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, sys, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path 
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Import project modules
# -------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Ann.builder import ANNBuilder
from src.AnnPso.annPsoTrainer import ANNPSOTrainer
# -------------------------------------------------------------------

# ==========================================================
# Utility helpers
# ==========================================================
def load_and_preprocess_data(path="Concrete_Data.xls", test_size=0.3, seed=42):
    if path is None:
        # go up one level to project root, then into Dataset
        path = Path(__file__).resolve().parent.parent / "Dataset" / "Concrete_Data.xls"
    df = pd.read_excel(path)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    print(f"Loaded dataset ({len(df)} samples) → Train {len(Xtr)} / Test {len(Xte)}")
    return Xtr, Xte, ytr, yte


def run_avg(trainer_fn, runs=10):
    """Repeat experiment multiple times and return mean±std of test MSE."""
    mses = []
    for i in range(runs):
        np.random.seed(i)
        mses.append(trainer_fn())
    return float(np.mean(mses)), float(np.std(mses))


def plot_bar(results, title, xlabel, ylabel, filename):
    os.makedirs("results/plots", exist_ok=True)
    labels, means, stds = zip(*results)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color="skyblue")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout(); fig.savefig(f"results/plots/{filename}", dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved → results/plots/{filename}")

# ==========================================================
# 1️⃣ Architecture investigation
# ==========================================================
def experiment_architecture(Xtr, Xte, ytr, yte, runs=10):
    print("\nEXPERIMENT 1: Architecture Comparison (10-run avg)")
    configs = [([5],"8-5-1"),([10],"8-10-1"),([20],"8-20-1"),
               ([10,5],"8-10-5-1"),([20,10],"8-20-10-1")]
    results=[]
    for layers,label in configs:
        def one_run():
            net=ANNBuilder.build_regression_network(8,layers)
            tr=ANNPSOTrainer(net,Xtr,ytr,Xte,yte,metric="mse")
            tr.train(swarm_size=30,max_iters=100,seed=np.random.randint(1e6),verbose=False)
            return tr.evaluate(Xte,yte)['mse']
        mean,std=run_avg(one_run,runs)
        print(f"{label:12s}  MSE={mean:.3f}±{std:.3f}")
        results.append((label,mean,std))
    plot_bar(results,"Effect of ANN Architecture","Architecture","Test MSE","arch_effect.png")
    return results

# ==========================================================
# 2️⃣ Swarm vs Iteration allocation
# ==========================================================
def experiment_swarm_iter(Xtr,Xte,ytr,yte,runs=10):
    print("\nEXPERIMENT 2: Swarm × Iterations (10-run avg)")
    configs=[(10,100),(20,50),(50,20),(100,10)]
    results=[]
    for s,it in configs:
        def one_run():
            net=ANNBuilder.build_regression_network(8,[10])
            tr=ANNPSOTrainer(net,Xtr,ytr,Xte,yte,metric="mse")
            tr.train(swarm_size=s,max_iters=it,seed=np.random.randint(1e6),verbose=False)
            return tr.evaluate(Xte,yte)['mse']
        mean,std=run_avg(one_run,runs)
        label=f"S{s}×I{it}"
        print(f"{label:10s}  {mean:.3f}±{std:.3f}")
        results.append((label,mean,std))
    plot_bar(results,"Swarm × Iterations (Budget = 1000)","Config","Test MSE","swarm_iter.png")
    return results

# ==========================================================
# 3️⃣ Acceleration coefficients c1 / c2
# ==========================================================
def experiment_c1c2(Xtr,Xte,ytr,yte,runs=10):
    print("\nEXPERIMENT 3: Acceleration Coefficients (10-run avg)")
    configs=[(2.5,0.5,"Exploration"),(2,1,"Cognitive"),(1.5,1.5,"Balanced"),
             (1,2,"Social"),(0.5,2.5,"Exploitation")]
    results=[]
    for c1,c2,lab in configs:
        def one_run():
            net=ANNBuilder.build_regression_network(8,[10])
            tr=ANNPSOTrainer(net,Xtr,ytr,Xte,yte,metric="mse")
            tr.train(swarm_size=30,max_iters=100,c1=c1,c2=c2,
                     seed=np.random.randint(1e6),verbose=False)
            return tr.evaluate(Xte,yte)['mse']
        mean,std=run_avg(one_run,runs)
        print(f"c1={c1:3.1f}, c2={c2:3.1f} → {mean:.3f}±{std:.3f}")
        results.append((lab,mean,std))
    plot_bar(results,"Effect of c₁ and c₂","Strategy","Test MSE","c1c2_effect.png")
    return results

# ==========================================================
# 4️⃣ PSO Topology
# ==========================================================
def experiment_topology(Xtr,Xte,ytr,yte,runs=10):
    print("\nEXPERIMENT 4: PSO Topology (10-run avg)")
    topologies=["random","ring","fully_informed"]
    results=[]
    for topo in topologies:
        def one_run():
            net=ANNBuilder.build_regression_network(8,[10])
            tr=ANNPSOTrainer(net,Xtr,ytr,Xte,yte,metric="mse")
            tr.train(swarm_size=30,max_iters=80,topology=topo,
                     seed=np.random.randint(1e6),verbose=False)
            return tr.evaluate(Xte,yte)['mse']
        mean,std=run_avg(one_run,runs)
        print(f"{topo:15s} {mean:.3f}±{std:.3f}")
        results.append((topo,mean,std))
    plot_bar(results,"Effect of PSO Topology","Topology","Test MSE","topology.png")
    return results

# ==========================================================
# 5️⃣ Design decisions – Activation + Boundary Handling
# ==========================================================
def experiment_design_decisions(Xtr, Xte, ytr, yte, runs=10):
    print("\nEXPERIMENT 5: Design Decisions (Activation × Boundary)")
    activations = ["relu", "tanh", "sigmoid"]
    boundaries = ["clamp", "reflect"]
    results = []

    # detect correct argument name
    arg_name = "activations"
    if "activation" in ANNBuilder.build_regression_network.__code__.co_varnames:
        arg_name = "activation"

    for act in activations:
        for b in boundaries:
            def one_run():
                # ✅ pass activation as list
                kwargs = {arg_name: [act]}
                net = ANNBuilder.build_regression_network(8, [10], **kwargs)
                tr = ANNPSOTrainer(net, Xtr, ytr, Xte, yte, metric="mse")
                tr.train(swarm_size=25, max_iters=80, boundary_mode=b,
                         seed=np.random.randint(1e6), verbose=False)
                return tr.evaluate(Xte, yte)['mse']

            mean, std = run_avg(one_run, runs)
            label = f"{act}-{b}"
            print(f"{label:15s} {mean:.3f}±{std:.3f}")
            results.append((label, mean, std))

    plot_bar(results,
             "Design Decisions (Activation × Boundary)",
             "Setting",
             "Test MSE",
             "design_decisions.png")

    return results

# ==========================================================
# Main runner
# ==========================================================
def main():
    os.makedirs("results",exist_ok=True)
    Xtr,Xte,ytr,yte=load_and_preprocess_data()
    allres={}
    allres["architecture"]=experiment_architecture(Xtr,Xte,ytr,yte)
    allres["swarm_iter"]=experiment_swarm_iter(Xtr,Xte,ytr,yte)
    allres["c1c2"]=experiment_c1c2(Xtr,Xte,ytr,yte)
    allres["topology"]=experiment_topology(Xtr,Xte,ytr,yte)
    allres["design"]=experiment_design_decisions(Xtr,Xte,ytr,yte)
    pd.concat({k:pd.DataFrame(v,columns=["Config","MeanMSE","Std"])
               for k,v in allres.items()}).to_csv("results/task5_summary.csv")
    print("\n✅ All experiments complete → results/task5_summary.csv\n")

if __name__=="__main__":
    main()
