# Databricks notebook source
import urllib
urllib.urlretrieve("https://resources.lendingclub.com/LoanStats3a.csv.zip", "/tmp/LoanStats3a.csv.zip")
dbutils.fs.mv("file:/tmp/LoanStats3a.csv.zip", "dbfs:/tmp/sample_zip/LoanStats3a.csv.zip")
display(dbutils.fs.ls("dbfs:/tmp/sample_zip"))
