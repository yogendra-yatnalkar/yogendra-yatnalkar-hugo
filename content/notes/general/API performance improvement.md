---
title: "Web-API performance improvement"
weight: 10
# bookFlatSection: false
# bookToc: true
# bookHidden: false
bookCollapseSection: false
bookComments: true
# bookSearchExclude: false
---

# API Performance Improvement

Based on: [Top 7 Ways to 10x Your API Performance - YouTube](https://www.youtube.com/watch?v=zvWKqUiovAM) 

Optimization should not be the first step of development

![](API%20performance%20improvement/2023-07-30-00-05-16-image.png)

## 1. Caching:

- If same request is repeated multiple times --> cache hence no need to recompute or hit the DB again.

- For DB, its: MemCacheD or Redis

## 2. Connection Pooling:

- ![](API%20performance%20improvement/2023-07-29-23-43-59-image.png)

- Having continues connections with DB can slow down server as each connection requires a lot of handshake protocol.  **Hence, it s a good practice to already have a set of connections ready with each set of API. This is difficult in serverless applications like Lambda and can cause big problems:**

- ![](API%20performance%20improvement/2023-07-29-23-47-20-image.png)

- Solution (at-least on AWs): **RDS Proxy:-** It sits between DB and applications (including AWS Lambda) to efficiently manage the DB connections

## 3. N+1 query problem:

- Ideally, we should fetch data in a single request to Db instead of asking or querying it N times. Conclusion being, we should try to club requests to query our DB. 

## 4. Pagination:

- If data to be fetched or requested from DB or server is huge, it will slow the response time --> Hence we should paginate our response into multiple pages to reduce data transfer in single go. 
  
  ![](API%20performance%20improvement/2023-07-29-23-53-53-image.png)

## 5. Json Serialization:

- Serialization takes time ...hence consider ways to reduce that time

- Example: Can think of using **gRPC** or some json serialization library which is very fast. 

## 6. Compress API response payloads to reduce network latency.

- [GitHub - google/brotli: Brotli compression format](https://github.com/google/brotli)

- Various CDN also perform these tasks example: **Cloudfare**

## 7. Async Logging:

- Logging is important but writing logs during stream processing applications can cause bottleneck 

- Hence, in such scenarios it is better to log logs via async operations. 

- But, there is a small chance that the some logs can be missed in this case. 

![](API%20performance%20improvement/2023-07-30-00-04-11-image.png)
