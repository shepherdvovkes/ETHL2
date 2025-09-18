package com.defimon.common.enums;

/**
 * Risk levels for cryptocurrency assets
 */
public enum RiskLevel {
    VERY_LOW("Very Low", 1),
    LOW("Low", 2),
    MEDIUM("Medium", 3),
    HIGH("High", 4),
    VERY_HIGH("Very High", 5),
    EXTREME("Extreme", 6);
    
    private final String displayName;
    private final int numericValue;
    
    RiskLevel(String displayName, int numericValue) {
        this.displayName = displayName;
        this.numericValue = numericValue;
    }
    
    public String getDisplayName() {
        return displayName;
    }
    
    public int getNumericValue() {
        return numericValue;
    }
    
    /**
     * Get risk level from numeric value
     */
    public static RiskLevel fromNumericValue(int value) {
        for (RiskLevel level : values()) {
            if (level.numericValue == value) {
                return level;
            }
        }
        throw new IllegalArgumentException("Invalid risk level numeric value: " + value);
    }
    
    /**
     * Get risk level from display name
     */
    public static RiskLevel fromDisplayName(String displayName) {
        for (RiskLevel level : values()) {
            if (level.displayName.equalsIgnoreCase(displayName)) {
                return level;
            }
        }
        throw new IllegalArgumentException("Invalid risk level display name: " + displayName);
    }
    
    /**
     * Check if this risk level is higher than another
     */
    public boolean isHigherThan(RiskLevel other) {
        return this.numericValue > other.numericValue;
    }
    
    /**
     * Check if this risk level is lower than another
     */
    public boolean isLowerThan(RiskLevel other) {
        return this.numericValue < other.numericValue;
    }
}
