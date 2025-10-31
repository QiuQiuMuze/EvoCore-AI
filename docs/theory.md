# EvoCore Runtime Mechanics (Resource-Focused Build)

This note documents what the current repository implements today. The runtime centres on resource discovery; there are no viral or "hack" subsystems in this branch. Index \(i\) denotes a living cell, \(p\) a parent, and \(m\) an optional memory exemplar drawn from a local pool.

## 1. Energy accounting
At every scheduler tick the engine updates cell energy by aggregating event-specific bonuses and penalties rather than a single metabolic subtraction. The update can be summarised as
\begin{equation}
\label{eq:energy-update}
E_i(t+1) = E_i(t) + \Delta_i^{+}(t) - \Delta_i^{-}(t),
\end{equation}
where \(\Delta_i^{+}\) collects all additive sources and \(\Delta_i^{-}\) all decrements observed in the code base:

* **Warm-up stipend.** For \(t < \texttt{warmup\_steps}\) every unit receives \(\texttt{warmup\_increment}\).【F:coggraph.py†L1333-L1343】【F:energy_policy.py†L46-L56】
* **High-frequency bonus.** Units with average call frequency above \(\texttt{high\_freq\_call\_threshold}\) add \(\texttt{high\_freq\_bonus}\).【F:CogUnit.py†L503-L521】【F:energy_policy.py†L58-L66】
* **Self-evaluation reward.** Each `CogUnit` earns \(\texttt{self\_reward\_scale} \cdot r^{\text{self}}_i(t)\) when its internal heuristic is positive.【F:CogUnit.py†L530-L548】【F:energy_policy.py†L68-L76】
* **Intrinsic goal completion.** An emitter that predicts its personal goal exactly gains \(\texttt{intrinsic\_goal\_bonus}\).【F:coggraph.py†L2473-L2497】【F:energy_policy.py†L78-L80】
* **Hazard escape.** Leaving a confirmed hazard grants \(\texttt{hazard\_escape\_bonus}\), dynamically scaled by recent danger rates.【F:coggraph.py†L2578-L2587】【F:energy_policy.py†L82-L123】
* **Resource proximity reward.** If an emitter predicts a resource with squared-error distance \(d\), it gains
  \begin{equation}
  \label{eq:resource-base}
  \Delta^{\text{res}}_i(t) = \texttt{resource\_base\_reward}(d),
  \end{equation}
  and each upstream processor \(j\) receives \(\texttt{resource\_upstream\_share} \times \Delta^{\text{res}}_i(t)\).【F:coggraph.py†L2588-L2624】【F:energy_policy.py†L125-L157】
* **Resource capture bonus.** A direct hit yields \(\texttt{resource\_hit\_bonus}\) for the emitter plus the same upstream share multiplier.【F:coggraph.py†L2625-L2654】【F:energy_policy.py†L125-L146】
* **Pool redistribution.** The central energy pool distributes a budget \(B\) across deficit cells via
  \begin{equation}
  \label{eq:pool}
  \Delta^{\text{pool}}_i(t) = \min\Bigl(\texttt{cap},\ \texttt{gap}_i(t),\ \frac{w_i}{\sum_j w_j}\,B\Bigr),
  \end{equation}
  where weights \(w_i\) incorporate activity, gene bias, and role-specific boosts.【F:coggraph.py†L1629-L1689】
* **Split bonuses.** Forced splits provide \(0.35\) energy to parent and child during the initial 2000 steps.【F:coggraph.py†L1553-L1562】

Decrements \(\Delta_i^{-}\) arise from: linger penalties when emitters stay near spent targets, diversity penalties when the group collapses to one action, inactivity decay, movement costs, progressive energy tax beyond the global cap, and mandatory pool deposits when overcharged cells cannot split.【F:coggraph.py†L2655-L2707】【F:coggraph.py†L1389-L1426】【F:coggraph.py†L1522-L1551】 No explicit compute-cost term or global supply injection variable exists in this implementation.

## 2. Inheritance and local memory reuse
Two mechanisms redistribute resources when cells retire or replicate:

1. **Energy inheritance.** When a unit dies of age with residual energy, the total for its role is equally shared across living heirs of the same role younger than 240 steps:
   \begin{equation}
   \label{eq:inheritance-energy}
   E_{\text{heir}} \gets E_{\text{heir}} + \frac{\sum_{k \in \mathcal{D}_r} E_k}{|\mathcal{H}_r|}.
   \end{equation}
   【F:coggraph.py†L438-L458】
2. **Clone initialisation.** A clone copies the parent's network and gene, then optionally fuses the top local memory sample if available:
   \begin{equation}
   \label{eq:gene-blend}
   G_{\text{child}} = 0.7\,G_p + 0.3\,G_m,
   \end{equation}
   and blends stored outputs with the same 0.6/0.4 weighting, while energy is partitioned \(0.6\) to the child and \(0.4\) retained by the parent.【F:CogUnit.py†L905-L1048】 No logistic scheduling or cosine weighting is used.

## 3. Reward routing for resource discoveries
Resource rewards route deterministically rather than through a probabilistic attribution scheme. Combining \eqref{eq:resource-base} with the hit bonus gives the emitter update
\begin{equation}
\label{eq:resource-total}
\Delta E^{\text{emit}} = \Delta^{\text{res}}_i(t) + \mathbb{1}[\text{hit}]\,\texttt{resource\_hit\_bonus},
\end{equation}
while each upstream processor receives
\begin{equation}
\label{eq:resource-share}
\Delta E^{\text{proc}} = \texttt{resource\_upstream\_share} \times \Delta E^{\text{emit}}.
\end{equation}
The share ratio adapts to recent success and role imbalance via `EnergyPolicy.update_environment_feedback`, but the distribution remains a fixed proportion for every connected processor; there is no softmax normalisation or additional credit terms.【F:coggraph.py†L2588-L2654】【F:energy_policy.py†L125-L180】

## 4. Policy optimisation and entropy regularisation
The learning agent trains a transformer policy with PPO or REINFORCE. The loss combines policy, value, and entropy terms with a constant coefficient:
\begin{equation}
\label{eq:rl-loss}
\mathcal{L} = \mathcal{L}_{\text{policy}} + \texttt{value\_coef}\,\mathcal{L}_{\text{value}} - \texttt{entropy\_coef}\,\mathbb{E}[\mathcal{H}(\pi)].
\end{equation}
There is no automated temperature schedule; `entropy_coef` is user-specified and fixed during training.【F:agents/rl_agent.py†L240-L287】 The `TransformerPolicyNetwork` itself only emits logits; stochastic sampling arises from wrapping the logits in a categorical distribution at action selection time.【F:agents/rl_agent.py†L116-L138】

## 5. Goal representation channels
Emitters track personal goals using three one-hot layers:
\begin{equation}
\label{eq:goal-vec}
\mathbf{g} = (g_{\text{res}},\ g_{\text{haz}},\ g_{\text{cur}}),
\end{equation}
corresponding to resources, hazards, and curiosity beacons. Goals are reassigned after resource captures or environmental updates, but no "hack" objective exists in this branch, keeping the system focused on resource discovery and hazard avoidance.【F:coggraph.py†L2404-L2457】
