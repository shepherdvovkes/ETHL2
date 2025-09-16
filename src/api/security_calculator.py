"""
Security Score Calculator

This module provides functions to calculate composite security scores
based on individual security metrics for crypto assets and L2 networks.
"""

from typing import Dict, Any, Optional
import math


class SecurityScoreCalculator:
    """Calculator for composite security scores"""
    
    @staticmethod
    def calculate_asset_security_score(security_metrics: Dict[str, Any]) -> float:
        """
        Calculate composite security score for a crypto asset
        
        Scoring breakdown:
        - Audit & Verification (30%): audit_status, audit_score, contract_verified
        - Smart Contract Security (25%): vulnerability_score, security features
        - Decentralization (20%): governance, validator distribution
        - Security Features (15%): multisig, timelock, emergency pause
        - Code Quality (10%): source code availability, protection mechanisms
        
        Args:
            security_metrics: Dictionary containing security metrics
            
        Returns:
            Composite security score (0-100)
        """
        if not security_metrics:
            return 0.0
        
        score = 0.0
        
        # 1. Audit & Verification (30 points max)
        audit_score = 0.0
        
        # Audit status
        audit_status = security_metrics.get("audit_status")
        if audit_status == "audited":
            audit_score += 15
        elif audit_status == "pending":
            audit_score += 5
        
        # Audit score (if available)
        audit_score_value = security_metrics.get("audit_score")
        if audit_score_value is not None:
            # Normalize audit score (assuming 0-10 scale)
            audit_score += min(audit_score_value * 1.5, 10)
        
        # Contract verification
        if security_metrics.get("contract_verified", False):
            audit_score += 5
        
        score += min(audit_score, 30)
        
        # 2. Smart Contract Security (25 points max)
        contract_score = 0.0
        
        # Vulnerability score (inverse - lower is better)
        vulnerability_score = security_metrics.get("vulnerability_score", 10)
        if vulnerability_score is not None:
            # Convert to positive score (0-10 scale, lower vulnerability = higher score)
            contract_score += max(0, 10 - vulnerability_score)
        
        # Security features
        security_features = [
            "reentrancy_protection",
            "overflow_protection", 
            "access_control",
            "pause_functionality"
        ]
        
        for feature in security_features:
            if security_metrics.get(feature, False):
                contract_score += 3.75  # 15 points / 4 features
        
        score += min(contract_score, 25)
        
        # 3. Decentralization (20 points max)
        decentralization_score = 0.0
        
        # Governance decentralization
        governance_score = security_metrics.get("governance_decentralization", 0)
        if governance_score is not None:
            decentralization_score += min(governance_score, 10)
        
        # Validator distribution
        validator_score = security_metrics.get("validator_distribution", 0)
        if validator_score is not None:
            decentralization_score += min(validator_score, 10)
        
        score += min(decentralization_score, 20)
        
        # 4. Security Features (15 points max)
        features_score = 0.0
        
        # Multisig wallets
        if security_metrics.get("multisig_wallets", False):
            features_score += 5
        
        # Timelock mechanisms
        if security_metrics.get("timelock_mechanisms", False):
            features_score += 5
        
        # Emergency pause
        if security_metrics.get("emergency_pause", False):
            features_score += 5
        
        score += min(features_score, 15)
        
        # 5. Code Quality (10 points max)
        code_score = 0.0
        
        # Source code availability
        if security_metrics.get("source_code_available", False):
            code_score += 5
        
        # Upgrade mechanisms (more secure = better)
        upgrade_mechanism = security_metrics.get("upgrade_mechanisms")
        if upgrade_mechanism == "governance":
            code_score += 5
        elif upgrade_mechanism == "proxy":
            code_score += 3
        elif upgrade_mechanism == "none":
            code_score += 1
        
        score += min(code_score, 10)
        
        return min(score, 100.0)
    
    @staticmethod
    def calculate_l2_security_score(security_metrics: Dict[str, Any]) -> float:
        """
        Calculate composite security score for an L2 network
        
        Scoring breakdown:
        - Audit & Verification (25%): audit_count, audit_firms, bug_bounty
        - Validator Security (25%): validator_count, slashing, multisig
        - Decentralization (20%): sequencer, governance, treasury
        - Smart Contract Security (20%): contract verification, security features
        - Network Security (10%): time to finality, upgrade mechanisms
        
        Args:
            security_metrics: Dictionary containing L2 security metrics
            
        Returns:
            Composite security score (0-100)
        """
        if not security_metrics:
            return 0.0
        
        score = 0.0
        
        # 1. Audit & Verification (25 points max)
        audit_score = 0.0
        
        # Audit count
        audit_count = security_metrics.get("audit_count", 0)
        if audit_count is not None:
            audit_score += min(audit_count * 3, 15)  # Max 15 points for audits
        
        # Bug bounty program
        if security_metrics.get("bug_bounty_program", False):
            audit_score += 5
        
        # Audit firms (quality indicator)
        audit_firms = security_metrics.get("audit_firms", [])
        if audit_firms and len(audit_firms) > 0:
            audit_score += min(len(audit_firms) * 2, 5)  # Max 5 points for multiple firms
        
        score += min(audit_score, 25)
        
        # 2. Validator Security (25 points max)
        validator_score = 0.0
        
        # Validator count
        validator_count = security_metrics.get("validator_count", 0)
        if validator_count is not None:
            # More validators = better security (logarithmic scale)
            validator_score += min(math.log10(max(validator_count, 1)) * 5, 15)
        
        # Slashing mechanism
        if security_metrics.get("slashing_mechanism", False):
            validator_score += 5
        
        # Multisig requirement
        if security_metrics.get("multisig_required", False):
            validator_score += 5
        
        score += min(validator_score, 25)
        
        # 3. Decentralization (20 points max)
        decentralization_score = 0.0
        
        # Sequencer decentralization
        sequencer_score = security_metrics.get("sequencer_decentralization", 0)
        if sequencer_score is not None:
            decentralization_score += min(sequencer_score, 10)
        
        # Governance decentralization
        governance_score = security_metrics.get("governance_decentralization", 0)
        if governance_score is not None:
            decentralization_score += min(governance_score, 10)
        
        score += min(decentralization_score, 20)
        
        # 4. Smart Contract Security (20 points max)
        contract_score = 0.0
        
        # Contract verification
        if security_metrics.get("contract_verified", False):
            contract_score += 5
        
        # Source code availability
        if security_metrics.get("source_code_available", False):
            contract_score += 5
        
        # Security features
        security_features = [
            "reentrancy_protection",
            "overflow_protection",
            "access_control", 
            "emergency_pause"
        ]
        
        for feature in security_features:
            if security_metrics.get(feature, False):
                contract_score += 2.5  # 10 points / 4 features
        
        score += min(contract_score, 20)
        
        # 5. Network Security (10 points max)
        network_score = 0.0
        
        # Time to finality (faster = better, but not too fast)
        time_to_finality = security_metrics.get("time_to_finality")
        if time_to_finality:
            # Parse time (assuming format like "2 minutes" or "1 hour")
            if "minute" in time_to_finality.lower():
                network_score += 8
            elif "hour" in time_to_finality.lower():
                network_score += 5
            elif "day" in time_to_finality.lower():
                network_score += 2
        
        # Upgrade mechanism
        upgrade_mechanism = security_metrics.get("upgrade_mechanism")
        if upgrade_mechanism:
            if "governance" in upgrade_mechanism.lower():
                network_score += 2
            elif "multisig" in upgrade_mechanism.lower():
                network_score += 1
        
        score += min(network_score, 10)
        
        return min(score, 100.0)
    
    @staticmethod
    def calculate_security_risk_score(security_metrics: Dict[str, Any]) -> float:
        """
        Calculate security risk score (inverse of security score)
        
        Args:
            security_metrics: Dictionary containing security metrics
            
        Returns:
            Risk score (0-100, where 100 is highest risk)
        """
        security_score = SecurityScoreCalculator.calculate_asset_security_score(security_metrics)
        return 100.0 - security_score
    
    @staticmethod
    def get_security_grade(security_score: float) -> str:
        """
        Convert security score to letter grade
        
        Args:
            security_score: Security score (0-100)
            
        Returns:
            Letter grade (A+, A, B+, B, C+, C, D, F)
        """
        if security_score >= 95:
            return "A+"
        elif security_score >= 90:
            return "A"
        elif security_score >= 85:
            return "B+"
        elif security_score >= 80:
            return "B"
        elif security_score >= 75:
            return "C+"
        elif security_score >= 70:
            return "C"
        elif security_score >= 60:
            return "D"
        else:
            return "F"
