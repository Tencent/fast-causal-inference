package com.tencent.weixin.utils;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.*;
import java.security.KeyStore;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

public class HttpUtils {
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public static OkHttpClient getUnsafeOkHttpClient() {
        OkHttpClient okHttpClient = new OkHttpClient.Builder()
                .readTimeout(60, TimeUnit.SECONDS)
                .connectTimeout(60, TimeUnit.SECONDS)
                .sslSocketFactory(SSLSocketClient.getSSLSocketFactory(), SSLSocketClient.getX509TrustManager())
                .hostnameVerifier(SSLSocketClient.getHostnameVerifier())
                .build();
        return okHttpClient;
    }

    public JsonObject call(String path, JsonObject params, HashMap<String, String> headers) {
        Gson gson = new Gson();
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");
        logger.info("request path = " + path + ", params=" + params.toString());
        RequestBody body = RequestBody.create(JSON, gson.toJson(params));


        Request.Builder builder = new Request.Builder().url(path);
        if (headers == null) {
            headers = new HashMap<>();
        }
        for (String key : headers.keySet()) {
            builder.addHeader(key, headers.get(key));
        }
        Request request = builder.post(body).build();
        try {
//            OkHttpClient client = new OkHttpClient();
            OkHttpClient client = getUnsafeOkHttpClient();
            Response response = client.newCall(request).execute();
            if (response.isSuccessful()) {
                String resBodyStr = response.body().string();
                //logger.info("response=" + resBodyStr);
                return JsonParser.parseString(resBodyStr).getAsJsonObject();
            } else {
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public JsonObject callGet(String path, HashMap<String, String> headers) {
//        OkHttpClient client = new OkHttpClient();
        logger.info("request path = " + path);
        OkHttpClient client = getUnsafeOkHttpClient();
        Request.Builder builder = new Request.Builder().url(path);
        for (String key : headers.keySet()) {
            builder.addHeader(key, headers.get(key));
        }
        Request request = builder.get().build();
        try {
            Response response = client.newCall(request).execute();
            if (response.isSuccessful()) {
                String resBodyStr = response.body().string();
                logger.info("response=" + resBodyStr);
                return JsonParser.parseString(resBodyStr).getAsJsonObject();
            } else {
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public JsonObject callGetRetry(String path, HashMap<String, String> headers, int retryNum) throws Exception {
        JsonObject resJsonObject = null;
        for (int i = 0; i < retryNum; i++) {
            resJsonObject = callGet(path, headers);
            if (resJsonObject != null) {
                break;
            } else {
                logger.info("retry_num=" + i);
            }
            Thread.sleep(1000);
        }
        return resJsonObject;
    }

    public JsonObject callRetry(String path, JsonObject params, HashMap<String, String> headers, int retryNum) throws Exception {
        JsonObject resJsonObject = null;
        for (int i = 0; i < retryNum; i++) {
            resJsonObject = call(path, params, headers);
            if (resJsonObject != null) {
                break;
            } else {
                logger.info("retry_num=" + i);
            }
            Thread.sleep(1000);
        }
        return resJsonObject;
    }
}


class SSLSocketClient {
    public static SSLSocketFactory getSSLSocketFactory() {
        try {
            SSLContext sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, getTrustManager(), new SecureRandom());
            return sslContext.getSocketFactory();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static TrustManager[] getTrustManager() {
        return new TrustManager[]{
                new X509TrustManager() {
                    @Override
                    public void checkClientTrusted(X509Certificate[] chain, String authType) {
                    }

                    @Override
                    public void checkServerTrusted(X509Certificate[] chain, String authType) {
                    }

                    @Override
                    public X509Certificate[] getAcceptedIssuers() {
                        return new X509Certificate[]{};
                    }
                }
        };
    }

    public static HostnameVerifier getHostnameVerifier() {
        return (s, sslSession) -> true;
    }

    public static X509TrustManager getX509TrustManager() {
        X509TrustManager trustManager = null;
        try {
            TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            trustManagerFactory.init((KeyStore) null);
            TrustManager[] trustManagers = trustManagerFactory.getTrustManagers();
            if (trustManagers.length != 1 || !(trustManagers[0] instanceof X509TrustManager)) {
                throw new IllegalStateException("Unexpected default trust managers:" + Arrays.toString(trustManagers));
            }
            trustManager = (X509TrustManager) trustManagers[0];
        } catch (Exception e) {
            e.printStackTrace();
        }

        return trustManager;
    }
}
