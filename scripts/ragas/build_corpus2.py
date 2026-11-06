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

    for f in sorted(out.iterdir()):
        print(f"{f.name:36s} {f.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
