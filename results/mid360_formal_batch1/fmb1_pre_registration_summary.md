# Mid-360 Formal Batch-1 预注册数据就绪总结

1. 实际 bag 数：36。
2. MAP/QUERY 对数：18，严格配对=true。
3. 3 Rich + 3 Weak 映射完整冻结=true。
4. SHA 认证：36/36。
5. acquisition PASS：18/18。
6. FAIL/REVIEW：FMB1_W02/None: SEMANTIC_CANDIDATE_GEOMETRY_CLASS_MISMATCH。
7. MAP 时长范围：[19.71976637840271, 19.813581228256226] s。
8. QUERY 时长范围：[14.72146487236023, 14.807947397232056] s。
9. gap 范围：[11.122812986373901, 11.431724309921265] s。
10. gap <10 s：0。
11. MAP/QUERY overlap：0。
12. LiDAR rate 范围：[9.999850647687873, 10.002626658608019] Hz。
13. IMU rate 范围：[199.74777584908003, 200.12122928163748] Hz。
14. 全部 livox_frame=true。
15. 全部必需字段齐全=true。
16. 明显运动 bag 数：0。
17. 需要补采 station 数：0。
18. 无 registration target：18/18。
19. QUERY 进入 target：0。
20. 每站 10 query=true。
21. snapshot 总数：180/180。
22. R candidates 整体 geometry-rich=true。
23. W candidates 整体 geometry-weak=false。
24. geometry PASS scenes：FMB1_R01, FMB1_R02, FMB1_R03, FMB1_W01, FMB1_W03。
25. REVIEW/REJECT scenes：FMB1_W02。
26. 最终 Rich scenes：3。
27. 最终 Weak scenes：2。
28. ICP 参数修改：否。
29. Open3D/PCL registration 执行：否。
30. NO_ICP_ATTESTATION PASS=true。
31. Formal Batch-1 verifier PASS=false（独立 verifier 运行后更新其独立报告，不回写冻结结果）。
32. 全量测试状态见最终执行日志；registration 执行类测试在 NO_FORMAL_REGISTRATION 模式下不得运行。
33. FMB1_PRE_REGISTRATION_DATA_READY=false。
34. 具备后续 360 次独立授权条件=false。

本冻结没有产生 T_est、误差、最终残差、fitness 或 solver result。
