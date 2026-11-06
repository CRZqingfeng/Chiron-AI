# -*- coding: utf-8 -*-
"""生成 RAGAS 第二组（异构语料）的 5 份文档。

为什么自己造而不是随便找：要**定向覆盖**已知弱项，而不是"多样"本身。
- .md  代码块/YAML 密集 → 测代码块切分
- .xlsx 多 sheet 表格 → 测表格按行切碎（第一组的已知弱项）
- .docx 纯散文 + 编号流程 → 测无表格结构的段落召回
- .txt  原始日志（无小节、时间戳噪声）→ 测无结构文本
- .pdf  分页 + 跨页表格 → 测 PDF 提取

场景：内部数据平台 Kestrel。五份文档共享同一套事实，故意让部分信息**重复**、
部分信息**只在某一份里**，这样跨文档/跨格式的题才有正确答案。

用法：python scripts/ragas/build_corpus2.py --out=.tmp/ragas/corpus2
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------- 唯一事实源（改这里，五份文档自动一致） ----------------
SERVICES_PROD = [
    ("gateway", "8080", "统一入口，Nginx 上游"),
    ("order-api", "18081", "订单读写"),
    ("user-api", "18082", "用户与鉴权"),
    ("report-worker", "18085", "报表异步任务"),
    ("monitor-agent", "18086", "指标采集与告警上报"),
]
NODES = [
    ("node-a", "16C / 32G", "gateway, order-api"),
    ("node-b", "16C / 32G", "gateway, order-api"),
    ("node-c", "8C / 16G", "order-api, report-worker（独占）"),
]
QUOTA = [
    ("gateway", "2C", "2G", "2"),
    ("order-api", "4C", "8G", "3"),
    ("user-api", "2C", "4G", "2"),
    ("report-worker", "8C", "16G", "1"),
    ("monitor-agent", "1C", "1G", "1"),
]


def write_md(p: Path) -> None:
    lines = [
        "# Kestrel 数据平台部署架构",
        "",
        "> 版本 v3.2 · 最后更新 2026-09-10 · 维护人：平台组",
        "",
        "## 1. 总体结构",
        "",
        "Kestrel 是一套面向内部业务的数据平台，对外只有一个入口 `gateway`，",
        "内部按职责拆成订单、用户、报表、监控四类服务。所有服务以容器方式运行在",
        "自建物理节点上，**不使用 Kubernetes**，扩缩容靠改部署脚本里的副本数后重发布。",
        "",
        "## 2. 服务清单与端口（生产环境）",
        "",
        "| 服务 | 端口 | 说明 |",
        "|---|---|---|",
    ]
    for name, port, desc in SERVICES_PROD:
        lines.append(f"| {name} | {port} | {desc} |")
    lines += [
        "",
        "非生产环境端口不同：dev 环境 order-api 用 8091、user-api 用 8092、report-worker 用 8095；",
        "test 环境沿用生产端口规则，具体以《服务端口与资源分配表》为准。",
        "",
        "## 3. 部署节点",
        "",
        "| 节点 | 规格 | 承载服务 |",
        "|---|---|---|",
    ]
    for n, spec, svc in NODES:
        lines.append(f"| {n} | {spec} | {svc} |")
    lines += [
        "",
        "report-worker 对内存敏感（单任务峰值约 12G），因此独占 node-c，不与 order-api 争抢。",
        "",
        "## 4. 启动方式",
        "",
        "每个服务都是独立进程，由 systemd 托管，启动脚本如下：",
        "",
        "```bash",
        "# /opt/kestrel/bin/start.sh",
        "export KESTREL_ENV=prod",
        "export KESTREL_CONFIG=/etc/kestrel/application.yaml",
        "",
        "nohup java -jar /opt/kestrel/order-api.jar \\",
        "  --server.port=${ORDER_API_PORT:-18081} \\",
        "  --spring.profiles.active=${KESTREL_ENV} \\",
        "  > /var/log/kestrel/order-api.log 2>&1 &",
        "```",
        "",
        "端口通过环境变量覆盖，未设置时回落到配置文件里的默认值。",
        "",
        "## 5. 关键配置片段",
        "",
        "```yaml",
        "kestrel:",
        "  gateway:",
        "    listen: 8080",
        "    upstream_timeout_ms: 3000",
        "  order:",
        "    db:",
        "      url: jdbc:mysql://10.20.30.11:3306/kestrel_order",
        "      pool_size: 20",
        "  cache:",
        "    redis:",
        "      host: 10.20.30.12",
        "      port: 6379",
        "      database: 3",
        "  storage:",
        "    minio:",
        "      endpoint: 10.20.30.13:9000",
        "      bucket: kestrel-report",
        "```",
        "",
        "## 6. 依赖关系",
        "",
        "1. `gateway` 依赖 `user-api` 做鉴权，鉴权失败直接 401，不再向下游转发。",
        "2. `order-api` 依赖 MySQL（10.20.30.11:3306）与 Redis（10.20.30.12:6379）。",
        "3. `report-worker` 依赖 MinIO（10.20.30.13:9000）保存报表文件，任务队列用 Redis 的 database 3。",
        "4. `monitor-agent` 只上报不参与业务链路，它挂掉不影响交易，但会失去告警能力。",
        "",
        "## 7. 变更纪律",
        "",
        "- 端口变更必须同步更新《服务端口与资源分配表》，否则下次发布极易撞端口。",
        "- 任何服务的新增/下线都要在本文件第 2 节登记，不允许「先上线后补文档」。",
        "- 遇到故障先看《故障应急响应手册》定级，不要在群里直接拉人。",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_xlsx(p: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    bold = Font(bold=True)

    def style_header(ws, ncol):
        for c in range(1, ncol + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = bold
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center")

    ws = wb.active
    ws.title = "端口分配"
    ws.append(["服务", "环境", "端口", "协议", "备注"])
    rows = [
        ("gateway", "prod", 8080, "HTTP", "对外唯一入口"),
        ("gateway", "test", 8080, "HTTP", "与 prod 同端口"),
        ("order-api", "dev", 8091, "HTTP", ""),
        ("order-api", "test", 18081, "HTTP", ""),
        ("order-api", "prod", 18081, "HTTP", "3 副本"),
        ("user-api", "dev", 8092, "HTTP", ""),
        ("user-api", "test", 18082, "HTTP", ""),
        ("user-api", "prod", 18082, "HTTP", "2 副本"),
        ("report-worker", "dev", 8095, "HTTP", ""),
        ("report-worker", "test", 18085, "HTTP", "⚠ 与 monitor-agent 冲突，见变更记录"),
        ("report-worker", "prod", 18085, "HTTP", "node-c 独占"),
        ("monitor-agent", "test", 18085, "HTTP", "⚠ 与 report-worker 冲突，待改 18087"),
        ("monitor-agent", "prod", 18086, "HTTP", "只上报"),
    ]
    for r in rows:
        ws.append(list(r))
    style_header(ws, 5)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 10
    ws.column_dimensions["E"].width = 46

    ws2 = wb.create_sheet("资源配额")
    ws2.append(["服务", "CPU", "内存", "副本数", "环境"])
    for name, cpu, mem, rep in QUOTA:
        ws2.append([name, cpu, mem, int(rep), "prod"])
    style_header(ws2, 5)
    for col, w in (("A", 18), ("B", 10), ("C", 10), ("D", 10), ("E", 10)):
        ws2.column_dimensions[col].width = w

    ws3 = wb.create_sheet("变更记录")
    ws3.append(["日期", "变更人", "变更内容", "状态"])
    changes = [
        ("2026-08-02", "zhangwei", "order-api prod 端口 18080 改为 18081，避开网关健康检查端口", "已完成"),
        ("2026-08-19", "liuyang", "report-worker test 环境启用 18085", "已完成"),
        ("2026-09-02", "liuyang", "monitor-agent test 环境也申请 18085，未登记冲突", "遗留问题"),
        ("2026-09-18", "liuyang", "report-worker test 端口临时改 18087 恢复服务", "已完成（临时）"),
        ("2026-09-25", "zhangwei", "monitor-agent test 端口规划改为 18088，尚未实施", "待办"),
    ]
    for c in changes:
        ws3.append(list(c))
    style_header(ws3, 4)
    for col, w in (("A", 14), ("B", 12), ("C", 62), ("D", 20)):
        ws3.column_dimensions[col].width = w

    wb.save(p)


def write_docx(p: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("Kestrel 数据平台故障应急响应手册", level=0)
    para = doc.add_paragraph("文档编号 OPS-KESTREL-004 · 版本 2.1 · 生效日期 2026-09-01")
    para.runs[0].font.size = Pt(9)

    doc.add_heading("一、适用范围", level=1)
    doc.add_paragraph(
        "本手册适用于 Kestrel 数据平台全部生产与测试环境。凡出现服务不可用、接口大量报错、"
        "数据不一致或资源告警，均按本手册处置。手册不覆盖机房断电、网络割接等基础设施级故障，"
        "那类故障直接升级至基础设施组。"
    )

    doc.add_heading("二、故障分级与时限", level=1)
    doc.add_paragraph(
        "P1 级：全站不可用，或已造成资金损失。要求 5 分钟内响应、30 分钟内恢复。"
        "定级后必须立即拉起应急群，成员包括当班值班长、平台负责人、受影响业务方负责人。"
    )
    doc.add_paragraph(
        "P2 级：核心功能受损但站点整体可用，例如下单成功率低于 90%。要求 15 分钟内响应、"
        "2 小时内恢复。需在群里同步进展，每 30 分钟一次。"
    )
    doc.add_paragraph(
        "P3 级：非核心功能异常，例如报表延迟、监控断点。要求 30 分钟内响应、1 个工作日内处理完毕。"
    )
    doc.add_paragraph(
        "P4 级：咨询、优化建议、文档纠错。1 个工作日内响应即可，不进应急群。"
    )
    doc.add_paragraph(
        "分级只升不降：处置过程中影响面扩大的，按更高级别重新定级并重新计时；不允许为了缩短时长而先定低级别。"
    )

    doc.add_heading("三、标准处置流程", level=1)
    for i, step in enumerate(
        [
            "确认现象与影响面。先看监控大盘的可用性与错误率曲线，再用日志确认报错关键字，不要凭感觉判断。",
            "定级。按第二节标准判定 P1 至 P4，并把判定依据写进应急群第一条消息。",
            "通知。P1 立即拉群；P2 通知值班长与平台负责人；P3 在工单系统登记即可。",
            "止损优先。能用回滚解决的一律先回滚，不要在故障期间做深度调优；必要时摘流、限流、降级。",
            "定位根因。按日志关键字、端口占用、资源水位三条线并行排查。",
            "复盘。P1 与 P2 必须在 48 小时内产出复盘报告，包含时间线、根因、改进项与责任人。",
        ],
        start=1,
    ):
        doc.add_paragraph(f"{i}. {step}")

    doc.add_heading("四、端口冲突的处置办法", level=1)
    doc.add_paragraph(
        "端口冲突是发布阶段最高频的故障。发现服务起不来且日志出现 address already in use 时，按以下顺序处理。"
    )
    doc.add_paragraph(
        "第一步，用 ss -lntp 加端口号确认占用进程，并确认它属于哪个服务、哪个环境。"
        "第二步，判断归属：如果占用方是同环境的另一个服务，原则是改后部署方（也就是本次要发布的服务）的端口，"
        "而不是去动已经在跑的服务。第三步，改完端口后必须同步更新《服务端口与资源分配表》，"
        "否则下一次发布还会撞。第四步，如果占用方是历史残留进程（启动时间早但无业务流量），"
        "可以直接终止后重启；但如果占用方属于其他团队，必须先通知对方负责人，严禁直接 kill。"
    )
    doc.add_paragraph(
        "临时改端口只用于恢复服务，事后必须补一次正式的端口规划，否则临时值会变成新的历史包袱。"
    )

    doc.add_heading("五、回滚步骤", level=1)
    doc.add_paragraph("任何发布引发的故障，优先执行回滚，回滚分三步，顺序不可颠倒。")
    doc.add_paragraph(
        "1. 切回上一版本的镜像 tag。当前版本镜像要保留，不要删除，便于事后分析。"
    )
    doc.add_paragraph(
        "2. 执行数据库回滚脚本。只有在本次发布包含 DDL 变更时才需要，纯代码发布跳过这一步。"
    )
    doc.add_paragraph(
        "3. 验证。先确认 /healthz 返回 200，再跑三条核心链路冒烟：下单、查单、取消单。"
        "三项全通过才算回滚完成。"
    )

    doc.add_heading("六、值班与联系人", level=1)
    doc.add_paragraph("值班表按周轮转，当班值班长对定级与拉群负最终责任。主要联系人如下。")
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "角色", "姓名", "负责范围"
    for role, name, scope in [
        ("平台负责人", "张伟", "架构决策、跨团队协调"),
        ("当班值班长", "轮值", "定级、拉群、对外同步"),
        ("订单域", "刘洋", "order-api、report-worker"),
        ("用户域", "陈静", "user-api、gateway"),
        ("监控", "赵磊", "monitor-agent、告警规则"),
    ]:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text = role, name, scope

    doc.save(p)


def write_txt(p: Path) -> None:
    lines = [
        "2026-09-18 10:11:52.331 INFO  [main] c.k.report.WorkerApplication - starting report-worker, env=test",
        "2026-09-18 10:11:52.334 INFO  [main] c.k.report.WorkerApplication - config loaded from /etc/kestrel/application.yaml",
        "2026-09-18 10:11:52.512 INFO  [main] o.s.b.w.embedded.tomcat.TomcatWebServer - Tomcat initialized with port(s): 18085 (http)",
        "2026-09-18 10:11:53.001 ERROR [main] o.s.b.w.embedded.tomcat.TomcatWebServer - Failed to start connector [Connector[HTTP/1.1-18085]]",
        "org.apache.catalina.LifecycleException: Protocol handler start failed",
        "  at org.apache.catalina.connector.Connector.startInternal(Connector.java:986)",
        "  at org.apache.catalina.util.LifecycleBase.start(LifecycleBase.java:171)",
        "Caused by: java.net.BindException: Address already in use",
        "  at sun.nio.ch.Net.bind0(Native Method)",
        "  at sun.nio.ch.Net.bind(Net.java:461)",
        "  at sun.nio.ch.ServerSocketChannelImpl.bind(ServerSocketChannelImpl.java:227)",
        "2026-09-18 10:11:53.004 WARN  [main] c.k.report.WorkerApplication - startup failed, retry 1/3 in 5s",
        "2026-09-18 10:11:58.112 INFO  [main] c.k.report.WorkerApplication - retrying startup",
        "2026-09-18 10:11:58.113 ERROR [main] o.s.b.w.embedded.tomcat.TomcatWebServer - Failed to start connector [Connector[HTTP/1.1-18085]]",
        "Caused by: java.net.BindException: Address already in use",
        "2026-09-18 10:12:03.220 WARN  [main] c.k.report.WorkerApplication - startup failed, retry 2/3 in 5s",
        "2026-09-18 10:12:08.331 ERROR [main] c.k.report.WorkerApplication - startup failed, retry 3/3 in 5s",
        "2026-09-18 10:12:13.442 ERROR [main] c.k.report.WorkerApplication - giving up after 3 attempts, exit code 1",
        "2026-09-18 10:12:13.500 INFO  [Thread-2] o.s.s.concurrent.ThreadPoolTaskExecutor - Shutting down ExecutorService 'applicationTaskExecutor'",
        "2026-09-18 10:12:13.501 INFO  [Thread-2] com.zaxxer.hikari.HikariDataSource - HikariPool-1 - Shutdown initiated",
        "2026-09-18 10:12:13.512 INFO  [Thread-2] com.zaxxer.hikari.HikariDataSource - HikariPool-1 - Shutdown completed",
        "2026-09-18 10:12:14.001 INFO  [sshd] connection from 10.20.30.44 port 51822",
        "2026-09-18 10:12:20.113 INFO  [shell] $ ss -lntp | grep 18085",
        "2026-09-18 10:12:20.114 INFO  [shell] LISTEN 0 128 *:18085 *:* users:((\"java\",pid=28741,fd=212))",
        "2026-09-18 10:12:24.220 INFO  [shell] $ ps -o pid,lstart,cmd -p 28741",
        "2026-09-18 10:12:24.221 INFO  [shell] 28741 Thu Sep 18 09:58:11 2026 /opt/kestrel/monitor-agent.jar --server.port=18085",
        "2026-09-18 10:12:31.003 INFO  [chat] liuyang: 占用方是 monitor-agent 的 test 实例，不是我这边",
        "2026-09-18 10:12:48.510 INFO  [chat] zhangwei: 按手册第四节，改后部署方的端口，不要去动已经在跑的监控",
        "2026-09-18 10:13:02.117 INFO  [shell] $ sed -i 's/18085/18087/' /etc/kestrel/application.yaml",
        "2026-09-18 10:13:10.220 INFO  [main] c.k.report.WorkerApplication - starting report-worker, env=test",
        "2026-09-18 10:13:10.221 INFO  [main] o.s.b.w.embedded.tomcat.TomcatWebServer - Tomcat initialized with port(s): 18087 (http)",
        "2026-09-18 10:13:12.884 INFO  [main] c.k.report.WorkerApplication - Started WorkerApplication in 2.663 seconds",
        "2026-09-18 10:13:12.885 INFO  [main] c.k.report.WorkerApplication - report-worker is up, listening on 18087",
        "2026-09-18 10:13:30.001 WARN  [pool-2-thread-1] c.k.report.ReportJobRunner - backlog=143 pending jobs, workers=1",
        "2026-09-18 10:14:02.442 WARN  [pool-2-thread-1] c.k.report.ReportJobRunner - backlog=151 pending jobs, workers=1",
        "2026-09-18 10:15:11.008 ERROR [pool-2-thread-1] c.k.report.ReportJobRunner - job 88213 failed: OOM while building sheet",
        "2026-09-18 10:15:11.014 ERROR [pool-2-thread-1] java.lang.OutOfMemoryError: Java heap space",
        "2026-09-18 10:15:12.771 INFO  [monitor] c.k.monitor.AlertPublisher - alert sent: node-c memory usage 94%",
        "2026-09-18 10:16:40.330 INFO  [chat] zhaolei: node-c 内存 94%，report-worker 独占这台，堆只给了 8G",
        "2026-09-18 10:17:02.115 INFO  [chat] zhangwei: 定级 P2，下单不受影响但报表全挂，按 P2 走",
        "2026-09-18 10:17:30.667 INFO  [chat] zhangwei: 应急群已建，拉刘洋、陈静、赵磊、业务方王琳",
        "2026-09-18 10:19:44.201 INFO  [shell] $ sed -i 's/-Xmx8g/-Xmx14g/' /opt/kestrel/bin/start-report.sh",
        "2026-09-18 10:20:02.883 INFO  [main] c.k.report.WorkerApplication - restarting with heap 14G",
        "2026-09-18 10:20:31.442 INFO  [pool-2-thread-1] c.k.report.ReportJobRunner - backlog draining, remaining=88",
        "2026-09-18 10:23:10.005 INFO  [pool-2-thread-1] c.k.report.ReportJobRunner - backlog draining, remaining=12",
        "2026-09-18 10:24:55.771 INFO  [pool-2-thread-1] c.k.report.ReportJobRunner - backlog cleared",
        "2026-09-18 10:25:11.220 INFO  [monitor] c.k.monitor.AlertPublisher - alert cleared: node-c memory usage 61%",
        "2026-09-18 10:25:30.118 INFO  [chat] zhangwei: 服务恢复，10:25 结束，总时长 13 分钟，P2 内",
        "2026-09-18 10:26:02.441 INFO  [chat] liuyang: 端口我临时用了 18087，正式规划我提个变更",
        "2026-09-18 10:26:20.005 WARN  [audit] config drift detected: report-worker port 18087 != registered 18085",
        "2026-09-18 10:31:00.000 INFO  [audit] follow-up ticket OPS-2291 created: 端口分配表待更新",
    ]
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_pdf(p: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1c", parent=styles["Heading1"], fontName="STSong-Light", fontSize=16, leading=22)
    h2 = ParagraphStyle("h2c", parent=styles["Heading2"], fontName="STSong-Light", fontSize=13, leading=18)
    body = ParagraphStyle("bodyc", parent=styles["BodyText"], fontName="STSong-Light", fontSize=10.5, leading=16)
    small = ParagraphStyle("smallc", parent=body, fontSize=9, textColor="#555555")

    doc = SimpleDocTemplate(
        str(p), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title="Kestrel API 速查",
    )
    flow = [
        Paragraph("Kestrel 数据平台 API 速查", h1),
        Paragraph("版本 1.4 · 适用 API 网关 2.x · 更新日期 2026-09-05", small),
        Spacer(1, 8),
        Paragraph(
            "本速查只列最常用的接口约定。完整字段定义以《订单接口详细设计》为准，"
            "本文不重复字段级说明，只给调用方最常踩的坑。", body),
        Spacer(1, 10),
        Paragraph("一、通用约定", h2),
        Paragraph(
            "所有业务接口以 /v1 开头，统一走网关 8080 端口。生产环境不得直连 order-api 的 18081，"
            "直连会绕过鉴权与限流。", body),
        Paragraph(
            "鉴权方式为 Bearer Token：请求头必须携带 Authorization: Bearer &lt;token&gt;。"
            "Token 由 user-api 签发，有效期 7200 秒，过期后需重新获取，不支持自动续期。", body),
        Paragraph("二、限流与配额", h2),
        Paragraph(
            "网关对单租户默认限流 100 QPS，突发允许到 150 QPS 但持续时间不超过 10 秒。"
            "超过限流返回 429，响应体带 Retry-After 秒数。报表类接口单独限流 20 QPS。", body),
        Paragraph("三、常用接口", h2),
    ]
    data = [
        ["方法", "路径", "说明"],
        ["POST", "/v1/orders", "创建订单，幂等键 X-Idempotency-Key"],
        ["GET", "/v1/orders/{id}", "查询订单详情"],
        ["POST", "/v1/orders/{id}/cancel", "取消订单，仅待支付与已支付可取消"],
        ["GET", "/v1/reports/{id}", "下载报表，返回 302 跳转 MinIO 预签名地址"],
    ]
    t = Table(data, colWidths=[18 * mm, 62 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), "#DDEBF7"),
        ("GRID", (0, 0), (-1, -1), 0.4, "#999999"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow += [t, Spacer(1, 10), Paragraph("四、错误码", h2)]
    err = [
        ["错误码", "含义", "处理建议"],
        ["40101", "Token 过期或无效", "重新获取 token，不要重试原请求"],
        ["40301", "租户无权访问该资源", "检查租户与订单归属"],
        ["40901", "库存不足", "提示用户，不要自动重试"],
        ["42901", "触发限流", "按 Retry-After 退避后重试"],
        ["50002", "下游 order-api 超时", "可重试一次，仍失败则转人工"],
    ]
    t2 = Table(err, colWidths=[22 * mm, 58 * mm, 80 * mm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), "#DDEBF7"),
        ("GRID", (0, 0), (-1, -1), 0.4, "#999999"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow += [t2, PageBreak(), Paragraph("五、调用示例", h2)]
    flow += [
        Paragraph(
            "创建订单推荐带幂等键，重复提交同一键返回首次结果而不是新建订单：", body),
        Spacer(1, 4),
        Paragraph(
            "curl -X POST http://gateway:8080/v1/orders \\<br/>"
            "  -H 'Authorization: Bearer $TOKEN' \\<br/>"
            "  -H 'X-Idempotency-Key: 6f1c9a2b' \\<br/>"
            "  -d '{\"sku\":\"A1001\",\"qty\":2}'", small),
        Spacer(1, 10),
        Paragraph("六、版本兼容", h2),
        Paragraph(
            "v1 承诺向后兼容，新增字段不影响老客户端；下线字段会提前两个版本在响应头 "
            "X-Deprecation 中告知。v2 尚在设计中，未对外开放。", body),
    ]
    doc.build(flow)


# ---------------- 第二批：把候选池从 40 chunk 撑到 100+ ----------------
# 目的不是"多"，而是让检索有真实干扰项：同一平台的其他主题文档。
# 事实必须与第一批一致（服务名、端口、配额、人物、事故编号），否则题目的参考答案会自相矛盾。

def write_order_api_docx(p: Path) -> None:
    """字段级接口设计：与《API 速查》呼应但更细，制造"相关但不含答案"的干扰。"""
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("订单服务接口详细设计", level=0)
    r = doc.add_paragraph("文档编号 DEV-ORDER-012 · 版本 3.6 · 更新 2026-09-08 · 作者 刘洋")
    r.runs[0].font.size = Pt(9)

    doc.add_heading("1. 概述", level=1)
    doc.add_paragraph(
        "本文只描述 order-api 的请求与响应字段级约定。鉴权、限流、错误码的统一约定见《Kestrel API 速查》，"
        "本文不重复。所有时间字段均为 ISO8601 带时区，金额单位为分，避免浮点误差。"
    )

    doc.add_heading("2. 创建订单 POST /v1/orders", level=1)
    doc.add_paragraph("请求体字段如下表。除标注可选的字段外均为必填，缺字段直接返回 400，不做默认值填充。")
    t = doc.add_table(rows=1, cols=4)
    t.style = "Table Grid"
    h = t.rows[0].cells
    h[0].text, h[1].text, h[2].text, h[3].text = "字段", "类型", "必填", "说明"
    for f, ty, req, desc in [
        ("sku", "string(32)", "是", "商品编码，需在商品中心存在"),
        ("qty", "int", "是", "数量，1 到 999"),
        ("tenant_id", "string(16)", "是", "租户标识，由网关注入，客户端传了也忽略"),
        ("channel", "string(16)", "否", "下单渠道，默认 app，可选 app / web / openapi"),
        ("remark", "string(200)", "否", "备注，不做业务校验"),
        ("idempotency_key", "string(64)", "否", "幂等键，也可放在请求头 X-Idempotency-Key"),
    ]:
        row = t.add_row().cells
        row[0].text, row[1].text, row[2].text, row[3].text = f, ty, req, desc

    doc.add_paragraph(
        "幂等实现：服务端以 tenant_id 加 idempotency_key 做唯一索引，24 小时内重复提交同一键直接返回首次结果，"
        "不会新建订单。没有传幂等键的请求不做去重，超时重试可能产生重复订单，客户端务必自己保证。"
    )

    doc.add_heading("3. 订单状态机", level=1)
    doc.add_paragraph(
        "订单只有五个状态：待支付、已支付、已发货、已完成、已取消。状态只能单向流转，不允许回退。"
        "待支付超过 30 分钟由定时任务自动关闭；已支付订单只能取消不能删除；已发货订单取消需要走售后流程。"
    )
    doc.add_paragraph(
        "取消接口 POST /v1/orders/{id}/cancel 只接受待支付与已支付两种状态，其他状态返回 40902。"
        "取消是异步的，接口返回 202 表示已受理，最终结果通过回调通知。"
    )

    doc.add_heading("4. 查询订单 GET /v1/orders/{id}", level=1)
    doc.add_paragraph(
        "返回体包含订单主体、明细行、支付信息三段。明细行最多返回 50 行，超过部分需要分页接口单独拉取。"
        "查询默认走从库，主从延迟最大 3 秒，下单后立刻查询可能读不到，需要强一致请带 ?consistent=true，"
        "但该参数会把请求打到主库，QPS 上限只有 50，不建议在高频链路使用。"
    )

    doc.add_heading("5. 回调与重试", level=1)
    doc.add_paragraph(
        "状态变更通过回调推送给业务方，回调地址在租户配置里登记。失败重试 5 次，间隔分别为 5 秒、30 秒、"
        "2 分钟、10 分钟、30 分钟，全部失败后进入死信队列，需要人工在后台重放。"
    )

    doc.add_heading("6. 已知限制", level=1)
    doc.add_paragraph(
        "单个订单最多 50 个明细行；单租户下单峰值按 100 QPS 设计，超过会被网关限流；"
        "跨租户查询一律拒绝，返回 40301。批量创建接口目前不提供，需要循环调用单个创建。"
    )
    doc.save(p)


def write_db_design_md(p: Path) -> None:
    lines = [
        "# 数据库设计说明（订单库）",
        "",
        "> 版本 v2.4 · 更新 2026-08-30 · 维护 刘洋",
        "",
        "## 1. 分库分表规则",
        "",
        "订单主库 `kestrel_order` 逻辑上是一张表，物理上按 `user_id % 8` 拆成 8 个分片，",
        "分片名 `t_order_0` 到 `t_order_7`，路由由中间件完成，应用不感知。",
        "",
        "分片键只能是 `user_id`。**按 `order_id` 直接查会全分片扫描**，因此所有按订单号查询的接口",
        "都必须带 `user_id`，否则走的是兜底广播查询，耗时是正常查询的 8 倍以上。",
        "",
        "## 2. 核心表结构",
        "",
        "```sql",
        "CREATE TABLE t_order_0 (",
        "  id            BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键',",
        "  order_no      VARCHAR(32)  NOT NULL COMMENT '业务订单号',",
        "  user_id       BIGINT       NOT NULL COMMENT '用户ID，分片键',",
        "  tenant_id     VARCHAR(16)  NOT NULL COMMENT '租户',",
        "  status        TINYINT      NOT NULL DEFAULT 0 COMMENT '0待支付 1已支付 2已发货 3已完成 4已取消',",
        "  amount        BIGINT       NOT NULL COMMENT '金额，单位分',",
        "  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,",
        "  updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,",
        "  PRIMARY KEY (id),",
        "  UNIQUE KEY uk_order_no (order_no),",
        "  UNIQUE KEY uk_tenant_idem (tenant_id, idempotency_key),",
        "  KEY idx_user_status (user_id, status, created_at)",
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "```",
        "",
        "`idempotency_key` 字段在示例里省略，实际与 `tenant_id` 组成唯一索引，这是幂等的实现基础。",
        "",
        "## 3. 索引约定",
        "",
        "- 所有查询必须命中索引，禁止 `select *` 无 `where` 的全表扫描。",
        "- `user_id + status + created_at` 是订单列表页的固定组合索引，顺序不可调整。",
        "- 新增索引前必须用 `EXPLAIN` 确认现有索引未命中，并在变更单里贴出执行计划。",
        "- 单表索引数量上限 5 个，超过说明查询模式有问题，应先收敛查询而不是继续加索引。",
        "",
        "## 4. 慢查询治理",
        "",
        "**注意：这一节讲的是治理流程，不包含 `EXPLAIN` 各字段的判读方法**，那部分见官方文档。",
        "",
        "慢查询阈值设为 1 秒，超过 1 秒的 SQL 会被记录到慢日志。每周导出一次并按出现次数排序，",
        "前 20 条必须逐一处理。处理方式优先是改查询、加索引次之，最后才考虑调优参数。",
        "",
        "## 5. 备份与恢复",
        "",
        "每日 02:00 全量备份，每小时 binlog 增量归档，保留 7 天。恢复演练每季度一次，",
        "演练目标是在 30 分钟内把单分片恢复到任意时间点。2026 年 Q3 演练结果：实测 22 分钟。",
        "",
        "## 6. 与其他组件的边界",
        "",
        "- Redis 只做缓存与任务队列，**不存业务主数据**，缓存失效后必须能从 MySQL 完整重建。",
        "- 报表数据落在 MinIO，不入库。",
        "- 监控指标走 monitor-agent 上报，不写在业务库里。",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_alert_rule_xlsx(p: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "告警规则"
    ws.append(["指标", "对象", "阈值", "持续时长", "级别", "通知渠道"])
    rows = [
        ("CPU 使用率", "全部节点", "85%", "5 分钟", "P3", "值班群"),
        ("内存使用率", "node-c", "90%", "3 分钟", "P2", "值班群 + 电话"),
        ("内存使用率", "node-a/node-b", "90%", "5 分钟", "P3", "值班群"),
        ("磁盘使用率", "全部节点", "80%", "10 分钟", "P3", "值班群"),
        ("下单成功率", "order-api", "<95%", "2 分钟", "P2", "值班群 + 负责人"),
        ("下单成功率", "order-api", "<80%", "1 分钟", "P1", "电话 + 应急群"),
        ("接口 P99 延迟", "gateway", ">1500ms", "5 分钟", "P3", "值班群"),
        ("接口错误率", "gateway", ">5%", "3 分钟", "P2", "值班群"),
        ("报表任务积压", "report-worker", ">100 条", "10 分钟", "P3", "值班群"),
        ("报表任务积压", "report-worker", ">500 条", "5 分钟", "P2", "值班群 + 负责人"),
        ("Redis 连接数", "10.20.30.12", ">800", "5 分钟", "P3", "值班群"),
        ("MySQL 主从延迟", "10.20.30.11", ">10 秒", "3 分钟", "P2", "值班群"),
    ]
    for r in rows:
        ws.append(list(r))
    for c in range(1, 7):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="FCE4D6")
        cell.alignment = Alignment(horizontal="center")
    for col, w in (("A", 18), ("B", 20), ("C", 12), ("D", 12), ("E", 8), ("F", 22)):
        ws.column_dimensions[col].width = w

    ws2 = wb.create_sheet("静默规则")
    ws2.append(["场景", "静默时长", "说明"])
    for r in [
        ("发布窗口", "30 分钟", "发布期间屏蔽 P3 级告警，P1/P2 不屏蔽"),
        ("备份窗口", "60 分钟", "每日 02:00 至 03:00 屏蔽磁盘 IO 相关告警"),
        ("演练窗口", "按需", "演练前在告警平台登记，登记期内只记录不通知"),
    ]:
        ws2.append(list(r))
    for c in range(1, 4):
        ws2.cell(row=1, column=c).font = Font(bold=True)
    for col, w in (("A", 18), ("B", 12), ("C", 56)):
        ws2.column_dimensions[col].width = w
    wb.save(p)


def write_capacity_docx(p: Path) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("2026 年 Q3 容量规划复盘", level=0)
    r = doc.add_paragraph("文档编号 CAP-2026Q3 · 编写 张伟 · 日期 2026-10-10")
    r.runs[0].font.size = Pt(9)

    doc.add_heading("一、Q3 实际水位", level=1)
    doc.add_paragraph(
        "订单峰值出现在 9 月 12 日，全天 186 万单，峰值 QPS 78，是容量设计值 100 QPS 的 78%。"
        "报表任务峰值出现在每月 1 日，单日 2.1 万个任务，report-worker 单副本平均耗时 4.2 分钟每任务。"
    )
    doc.add_paragraph(
        "节点水位方面，node-c 峰值内存占用 94%，已触发 P2 告警（见 9 月 18 日事故）；"
        "node-a 与 node-b 峰值内存 61%，仍有空间。CPU 三台节点均未超过 70%。"
    )

    doc.add_heading("二、Q4 扩容建议", level=1)
    doc.add_paragraph(
        "第一优先是把 node-c 从 8C16G 扩到 16C32G。理由是 report-worker 独占 node-c 且堆已调到 14G，"
        "现有 16G 物理内存只剩 2G 余量，再有任何驻留进程都会触发 OOM。扩到 32G 后堆可以给到 24G，"
        "报表积压消化能力预计提升 1.6 倍。"
    )
    doc.add_paragraph(
        "第二优先是给 report-worker 增加第二个副本，但需要先把任务队列的抢占逻辑改成分布式锁，"
        "否则两个副本会重复消费同一任务。这项工作预计 3 人日，建议放在 Q4 上半段。"
    )
    doc.add_paragraph(
        "暂不建议扩 order-api 副本数。当前 3 副本下 P99 延迟 210ms，离 1500ms 的告警线还很远，"
        "盲目扩容只会增加数据库连接数压力。"
    )

    doc.add_heading("三、成本与收益", level=1)
    doc.add_paragraph(
        "node-c 扩容一次性投入约 2.4 万元，年度电费与折旧摊到每月约 900 元。"
        "对应收益是报表类 P2 事故预计从每季度 1.5 次降到 0.3 次，按每次事故平均影响 40 分钟计算，"
        "一个季度可减少约 48 分钟的对外不可用时间。"
    )

    doc.add_heading("四、遗留问题", level=1)
    doc.add_paragraph(
        "端口分配表仍有未收口的冲突（monitor-agent test 环境 18085 与 report-worker 冲突，工单 OPS-2291），"
        "本季度未解决，顺延到 Q4。教训是端口申请必须走登记流程，口头约定一定会忘。"
    )
    doc.save(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/ragas/corpus2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    write_md(out / "Kestrel平台部署架构.md")
    write_xlsx(out / "服务端口与资源分配表.xlsx")
    write_docx(out / "故障应急响应手册.docx")
    write_txt(out / "2026-09-18_发布故障日志.txt")
    write_pdf(out / "Kestrel_API速查.pdf")
    # 第二批：撑大候选池
    write_order_api_docx(out / "订单服务接口详细设计.docx")
    write_db_design_md(out / "数据库设计说明.md")
    write_alert_rule_xlsx(out / "监控指标与告警规则.xlsx")
    write_capacity_docx(out / "2026Q3容量规划复盘.docx")

    for f in sorted(out.iterdir()):
        print(f"{f.name:36s} {f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
