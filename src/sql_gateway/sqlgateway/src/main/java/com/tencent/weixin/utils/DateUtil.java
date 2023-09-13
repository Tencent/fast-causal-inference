package com.tencent.weixin.utils;

import java.sql.Timestamp;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class DateUtil {
    public static String getTodayday(){
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");
        Calendar calendar = Calendar.getInstance();
        calendar.add(Calendar.DATE, 0);
        return dateFormat.format(calendar.getTime());
    }

    public static String getNowTime() {
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss");
        String dateString = formatter.format(currentTime);
        return dateString;
    }

    public static String getNowDay() {
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        String dateString = formatter.format(currentTime);
        return dateString;
    }

    public static String getNow() {
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String dateString = formatter.format(currentTime);
        return dateString;
    }

    public static String getDateBeforeDate(String dateStr) throws ParseException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");
        Date date = dateFormat.parse(dateStr);
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        cal.add(Calendar.DAY_OF_YEAR, -1);
        String targetDate = dateFormat.format(cal.getTime());
        return targetDate;
    }

    public static int getHour(String dateTimeStr) throws ParseException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        Date date = dateFormat.parse(dateTimeStr);
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        return cal.get(Calendar.HOUR_OF_DAY);
    }

    public static long getDay(String dateTimeStr) throws ParseException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        dateFormat.parse(dateTimeStr);
        String day = dateTimeStr.substring(0, 4)+ dateTimeStr.substring(5, 7) + dateTimeStr.substring(8, 10);
        return Long.parseLong(day);
    }

    public static String stampToTime(long timestamp) {
        String res;
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        Date date = new Date(timestamp * 1000);
        res = simpleDateFormat.format(date);
        return res;
    }

    public static String stampToDs(long timestamp) {
        String res;
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyyMMdd");
        Date date = new Date(timestamp * 1000);
        res = simpleDateFormat.format(date);
        return res;
    }

    public static Long diffTime(Timestamp timestamp1, Timestamp timestamp2) {
        long diffTime = Math.abs(timestamp2.getTime() - timestamp1.getTime());
        System.out.println(timestamp2.getTime());
        System.out.println(timestamp1.getTime());
        long minute = diffTime / 1000 / 60;
        return minute;
    }

    public static Long diffTime(Timestamp timestamp) {
        long diffTime = Math.abs(System.currentTimeMillis() - timestamp.getTime());
        long minute = diffTime / 1000 / 60;
        return minute;
    }
    
    public static void main(String[] args) {
        System.out.println(DateUtil.getNowTime());
    }

}
