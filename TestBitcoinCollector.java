import java.io.*;
import java.net.*;
import java.util.*;

/**
 * Simple Bitcoin Data Collector Test - Java 8
 * Tests QuickNode Bitcoin API integration
 */
public class TestBitcoinCollector {
    
    private static final String QUICKNODE_URL = "https://orbital-twilight-mansion.btc.quiknode.pro/a1280f4e959966b62d579978248263e3975e3b4d/";
    private static final String RPC_USER = "bitcoin";
    private static final String RPC_PASSWORD = "ultrafast_archive_node_2024";
    
    public static void main(String[] args) {
        System.out.println("=== Bitcoin Data Collector Test ===");
        System.out.println("Testing QuickNode Bitcoin API integration...");
        
        try {
            // Test 1: Get block count
            System.out.println("\n1. Testing getblockcount...");
            String blockCount = makeRpcCall("getblockcount");
            System.out.println("Current block count: " + blockCount);
            
            // Test 2: Get network info
            System.out.println("\n2. Testing getnetworkinfo...");
            String networkInfo = makeRpcCall("getnetworkinfo");
            System.out.println("Network info: " + networkInfo);
            
            // Test 3: Get mempool info
            System.out.println("\n3. Testing getmempoolinfo...");
            String mempoolInfo = makeRpcCall("getmempoolinfo");
            System.out.println("Mempool info: " + mempoolInfo);
            
            // Test 4: Get difficulty
            System.out.println("\n4. Testing getdifficulty...");
            String difficulty = makeRpcCall("getdifficulty");
            System.out.println("Difficulty: " + difficulty);
            
            System.out.println("\n=== All tests completed successfully! ===");
            
        } catch (Exception e) {
            System.err.println("Error during testing: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static String makeRpcCall(String method, Object... params) throws Exception {
        URL url = new URL(QUICKNODE_URL);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        
        // Set up the request
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setDoOutput(true);
        
        // Add basic authentication
        String auth = RPC_USER + ":" + RPC_PASSWORD;
        String encodedAuth = Base64.getEncoder().encodeToString(auth.getBytes());
        connection.setRequestProperty("Authorization", "Basic " + encodedAuth);
        
        // Create JSON-RPC request
        StringBuilder jsonRequest = new StringBuilder();
        jsonRequest.append("{");
        jsonRequest.append("\"jsonrpc\":\"1.0\",");
        jsonRequest.append("\"id\":\"test\",");
        jsonRequest.append("\"method\":\"").append(method).append("\"");
        
        if (params.length > 0) {
            jsonRequest.append(",\"params\":[");
            for (int i = 0; i < params.length; i++) {
                if (i > 0) jsonRequest.append(",");
                if (params[i] instanceof String) {
                    jsonRequest.append("\"").append(params[i]).append("\"");
                } else {
                    jsonRequest.append(params[i]);
                }
            }
            jsonRequest.append("]");
        }
        
        jsonRequest.append("}");
        
        // Send the request
        try (OutputStream os = connection.getOutputStream()) {
            byte[] input = jsonRequest.toString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        // Read the response
        int responseCode = connection.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                return response.toString();
            }
        } else {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(connection.getErrorStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                throw new Exception("HTTP " + responseCode + ": " + response.toString());
            }
        }
    }
}
