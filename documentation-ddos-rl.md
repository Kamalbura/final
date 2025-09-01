# FINAL CORRECTED DDoS-RL Agent Software Requirements Specification (SRS)
## Version 3.0 - Event-Driven with Proper Logic & 3-Action Space

---

## 1. PROJECT OVERVIEW

### 1.1 System Purpose
Develop a Reinforcement Learning-based agent that **outputs algorithm selection** to an existing scheduler, optimizing DDoS detection power consumption while protecting critical UAV systems (proxy script + MAVProxy) through intelligent resource management.

### 1.2 System Integration Architecture
```
Existing Scheduler (User Implemented)
    ↓
Network Monitoring → Threat_Response Change Detection
    ↓
RL Agent Triggered → Algorithm Selection Output
    ↓
Scheduler Executes Chosen Algorithm
```

**CRITICAL**: RL agent is **event-driven** - only activates when Threat_Response changes

---

## 2. FINALIZED SYSTEM SPECIFICATIONS

### 2.1 Action Space (3 Actions - Trinary Decision)

| Action ID | Algorithm | Power Impact | Use Case | Critical Priority |
|-----------|-----------|--------------|----------|-------------------|
| **0** | **No DDoS Algorithm** | Minimal | Emergency/Critical battery/thermal | Preserve proxy + MAVProxy |
| **1** | **XGBoost Detection** | Low (8.5ms, 12MB) | Normal operations, routine monitoring | Efficient security |
| **2** | **TST Detection** | High (45ms, 95MB) | Threat confirmation when justified | Maximum accuracy |

### 2.2 State Space (270 Discrete States)

| Dimension | Categories | Count | Discretization | Trigger Role |
|-----------|------------|-------|----------------|--------------|
| **Battery Level** | 0-20%, 21-40%, 41-60%, 61-80%, 81-100% | 5 | Continuous → Discrete | Context |
| **Threat Response** | Normal(0), Confirming(1), Confirmed(2) | 3 | Discrete categorical | **EVENT TRIGGER** |
| **Temperature** | Safe, Warning, Critical | 3 | Continuous → Discrete | Context |
| **Time Since Change** | 0-5s, 6-15s, 16-30s, 31-60s, 61-120s, 121-300s | 6 | Continuous → Discrete | Context |

**Total State Space**: 5 × 3 × 3 × 6 = **270 discrete states**

**CRITICAL**: RL agent only executes when **Threat_Response changes** (event-driven)

---

## 3. CORRECTED DECISION LOGIC

### 3.1 Power Optimization Principles (CORRECTED)

#### Rule 1: Normal Threats Never Justify TST
- **Normal State**: No threats detected
- **Maximum Action**: XGBoost monitoring (when battery allows)
- **Never Action 2**: TST execution is wasteful for normal traffic
- **Logic**: Why run expensive confirmation when no threat exists?

#### Rule 2: Critical System Priority
- **Battery < 20%**: Action 0 (No DDoS) - Preserve proxy + MAVProxy
- **Temperature > 80°C**: Action 0 (No DDoS) - Prevent thermal damage
- **System Overload**: Action 0 (No DDoS) - Maintain critical functions

#### Rule 3: Threat-Justified Resource Usage
- **Confirming Threats**: Consider TST based on resources
- **Confirmed Threats**: TST justified when resources allow
- **Resource Constraints**: Always protect critical systems first

### 3.2 Corrected Decision Matrix

| Battery Level | Normal Threat | Confirming Threat | Confirmed Threat |
|---------------|---------------|-------------------|------------------|
| **0-20%** (Critical) | **Action 0** (No DDoS) | **Action 0** (No DDoS) | **Action 0** (No DDoS) |
| **21-40%** (Low) | **Action 0** (No DDoS) | **Action 1** (XGBoost) | **Action 1** (XGBoost) |
| **41-60%** (Medium) | **Action 1** (XGBoost) | **Action 1** (XGBoost) | **Action 2** (TST) |
| **61-80%** (High) | **Action 1** (XGBoost) | **Action 2** (TST) | **Action 2** (TST) |
| **81-100%** (Full) | **Action 1** (XGBoost) | **Action 2** (TST) | **Action 2** (TST) |

**Key Corrections**:
- **Normal threats**: Never exceed Action 1 (XGBoost)
- **Critical battery**: Always Action 0 (protect critical systems)
- **TST reserved**: Only for actual threat scenarios with adequate resources

---

## 4. EVENT-DRIVEN OPERATION MODEL

### 4.1 RL Agent Activation
**Trigger Condition**: Threat_Response state change detected
**Activation Frequency**: Event-driven (not continuous)
**Decision Latency**: <1ms (Q-table lookup)
**Output**: Single integer (0, 1, or 2) to scheduler

### 4.2 System Integration Flow
```python
# In existing scheduler
def on_threat_response_change(new_threat_level):
    current_state = {
        'battery': get_battery_level(),
        'threat': new_threat_level,  # 0=Normal, 1=Confirming, 2=Confirmed
        'temperature': get_temperature(),
        'time_since_change': get_time_since_last_change()
    }
    
    # RL agent decides which algorithm to run
    algorithm_choice = rl_agent.select_algorithm(current_state)
    
    # Execute based on RL decision
    execute_chosen_algorithm(algorithm_choice)
```

### 4.3 Algorithm Execution Logic
- **Action 0**: Stop all DDoS algorithms, preserve critical systems
- **Action 1**: Run XGBoost detection only (lightweight)
- **Action 2**: Run TST detection (heavy confirmation)

---

## 5. CORRECTED PERFORMANCE EXPECTATIONS

### 5.1 Power Efficiency (Realistic)
**Normal Operations (70% of time)**:
- Action Distribution: 10% No DDoS, 90% XGBoost, 0% TST
- Power Efficiency: Excellent (no wasteful TST runs)
- Critical System Protection: Maintained

**Threat Scenarios (30% of time)**:
- Action Distribution: 20% No DDoS, 40% XGBoost, 40% TST
- Power Management: Resource-aware threat confirmation
- Security Effectiveness: Maintained through intelligent escalation

### 5.2 Resource Protection Validation
- **Battery < 20%**: 100% Action 0 (critical system protection)
- **Temperature > 80°C**: 100% Action 0 (thermal protection)
- **Normal Threats**: 0% TST execution (power optimization)
- **Confirmed Threats + Resources**: High TST usage (security priority)

---

## 6. IMPLEMENTATION ARCHITECTURE

### 6.1 RL Agent Interface (Final)
```python
class FinalDDoSRLAgent:
    def select_algorithm(self, battery_level: float, threat_response: int,
                        temperature: float, time_since_change: float) -> int:
        """
        Event-driven algorithm selection for existing scheduler
        
        Called ONLY when Threat_Response changes
        
        Args:
            battery_level: 0-100% battery remaining
            threat_response: 0=Normal, 1=Confirming, 2=Confirmed
            temperature: 30-85°C CPU temperature
            time_since_change: 0-300s since last algorithm change
            
        Returns:
            0 = No DDoS (preserve critical systems)
            1 = XGBoost (lightweight detection)
            2 = TST (heavy detection)
        """
        
        # Discretize state
        state_id = self.compute_state_id(battery_level, threat_response, 
                                        temperature, time_since_change)
        
        # Q-table lookup
        return self.q_table[state_id].argmax()
```

### 6.2 Decision Logic Implementation
```python
def corrected_decision_logic(battery_idx, threat_idx, temp_idx, time_idx):
    """Corrected logic ensuring normal threats never trigger TST"""
    
    # Critical system protection - always highest priority
    if battery_idx == 0 or temp_idx == 2:  # Battery <20% or temp >80°C
        return 0  # No DDoS - preserve critical systems
    
    # Normal threats - never justify expensive TST
    if threat_idx == 0:  # Normal (no threats)
        if battery_idx == 1:  # 21-40% battery
            return 0  # No DDoS - conserve for critical systems
        else:  # 41%+ battery
            return 1  # XGBoost - routine monitoring only
    
    # Confirming threats - resource-aware TST consideration
    elif threat_idx == 1:  # Confirming (potential threat)
        if battery_idx <= 1:  # Low battery
            return 1  # XGBoost - lightweight investigation
        elif temp_idx == 1 and battery_idx == 2:  # Warning temp + medium battery
            return 1  # XGBoost - thermal caution
        else:  # Adequate resources
            return 2  # TST - confirm potential threat
    
    # Confirmed threats - TST when resources permit
    elif threat_idx == 2:  # Confirmed (high confidence threat)
        if battery_idx <= 1:  # Low battery
            return 1  # XGBoost - best effort with constraints
        else:  # Adequate battery
            return 2  # TST - maximum accuracy for confirmed threats
    
    return 1  # Default to XGBoost
```

---

## 7. CORRECTED SUCCESS CRITERIA

### 7.1 Power Efficiency Targets (Realistic)
- **Normal Operations**: 0% TST execution (CORRECTED: no wasteful runs)
- **Critical Battery Protection**: 100% Action 0 when battery <20%
- **Thermal Protection**: 100% Action 0 when temperature >80°C
- **Overall Power Savings**: 40-60% vs. always-on TST approach

### 7.2 Security Effectiveness (Maintained)
- **Threat Detection**: >95% through XGBoost+TST intelligent escalation
- **Response Time**: <10s for confirmed threats (when resources allow)
- **False Positive Management**: Intelligent resource allocation prevents waste

### 7.3 System Protection (Critical)
- **Critical System Uptime**: 100% (proxy + MAVProxy never compromised)
- **Thermal Safety**: 100% protection through intelligent shutdown
- **Mission Completion**: Enhanced through intelligent resource management

---

## 8. IMPLEMENTATION READINESS

### 8.1 Integration Requirements
- **Input Interface**: Threat_Response change events from existing monitoring
- **Output Interface**: Algorithm selection (0, 1, or 2) to existing scheduler
- **State Monitoring**: Battery, temperature, timing from existing systems
- **Q-Table Storage**: 270 × 3 = 810 entries (~6.5KB memory)

### 8.2 Training and Validation
- **Training Scenarios**: Focus on realistic threat distributions
- **Validation**: Confirm 0% TST for normal threats
- **Testing**: Emergency scenarios protect critical systems
- **Performance**: Monitor power savings and security effectiveness

---

## 9. CONCLUSION

### 9.1 Critical Correction Achieved
**Thank you for identifying the fundamental logic flaw!** The corrected system ensures:

✅ **Normal threats NEVER trigger wasteful TST execution**  
✅ **Critical systems (proxy + MAVProxy) always protected**  
✅ **TST reserved for actual threat scenarios with adequate resources**  
✅ **Event-driven operation integrates cleanly with existing scheduler**  
✅ **3-action space provides complete operational flexibility**

- **Realistic Power Optimization**: No wasteful processing during normal operations
- **System Survival Priority**: Critical functions never compromised
- **Intelligent Resource Management**: Threat-justified algorithm escalation
- **Clean Integration**: Simple integer output to existing infrastructure

**The corrected DDoS-RL agent now represents a truly intelligent, power-efficient cybersecurity solution that respects system priorities and resource constraints.**

# Temperature-Aware Decision Logic for the DDoS-RL Agent

To understand how **temperature** impacts the RL agent’s algorithm selection, let’s incorporate thermal constraints into our final state–action matrix. 

## 1. Brainstorm: Temperature Impact Principles

1. **Safe (≤55°C)**  
   - No additional constraints—decisions follow the battery/threat rules.

2. **Warning (56–70°C)**  
   - Moderate caution—allow TST for battery ≥41%, but prefer XGBoost for 21–40%.

3. **Critical (>70°C)**  
   - Strict thermal protection—disable TST entirely, even for confirmed threats, to prevent overheating.

## 2. Revised Decision Matrix by Threat, Battery & Temperature

| Threat State   | Temperature | 0–20% Bat | 21–40% Bat | 41–60% Bat | 61–80% Bat | 81–100% Bat |
|----------------|-------------|-----------|------------|------------|------------|-------------|
| **Normal (0)**    | Safe       | No DDoS   | XGBoost    | XGBoost    | XGBoost    | XGBoost     |
|                  | Warning    | No DDoS   | XGBoost    | XGBoost    | XGBoost    | XGBoost     |
|                  | Critical   | No DDoS   | No DDoS    | No DDoS    | No DDoS    | No DDoS     |
| **Confirming (1)**| Safe       | No DDoS   | XGBoost    | **TST**    | **TST**    | **TST**     |
|                  | Warning    | No DDoS   | XGBoost    | **TST**    | **TST**    | **TST**     |
|                  | Critical   | No DDoS   | No DDoS    | No DDoS    | No DDoS    | No DDoS     |
| **Confirmed (2)** | Safe       | No DDoS   | XGBoost    | **TST**    | **TST**    | **TST**     |
|                  | Warning    | No DDoS   | XGBoost    | **TST**    | **TST**    | **TST**     |
|                  | Critical   | No DDoS   | No DDoS    | No DDoS    | No DDoS    | No DDoS     |

- **TST** is only used in **Safe** and **Warning** environments when battery ≥41% and a threat is present.
- **No DDoS** is enforced during **Critical** temperature for all threat/battery combinations.

## 3. Visualization: Heatmap for Confirming Threats

Below is a heatmap for the **Confirming (1)** threat state, showing how the RL agent selects between:
- **0 (No DDoS)** – red  
- **1 (XGBoost)** – yellow  
- **2 (TST)** – green  

across **battery levels** (rows) and **temperature bins** (columns).

![Temperature-Aware Decision Heatmap]

Matrix data for plotting:
```
Battery\Temp  Safe   Warning  Critical
0-20%         0      0        0
21-40%        1      1        0
41-60%        2      2        0
61-80%        2      2        0
81-100%       2      2        0
```

- **Rows**: Battery bins 0–20%, 21–40%, 41–60%, 61–80%, 81–100%
- **Columns**: Temperature bins Safe, Warning, Critical
- **Values**: Decision Action (0, 1, 2)

## 4. Next Steps

- **Incorporate** this temperature-aware logic into the final SRS and implementation code.
- **Validate** across field-tested thermal scenarios.
- **Benchmark** power savings and security effectiveness under Warning and Critical temperatures.

This completes our temperature‐aware refinement, ensuring the RL agent protects the hardware while still confirming genuine threats when conditions allow.
