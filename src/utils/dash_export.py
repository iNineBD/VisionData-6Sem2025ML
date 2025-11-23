from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from io import BytesIO


def generate_forecast_pdf(charts, output_file="relatorio_previsoes.pdf"):
    """Gera um PDF com total controle de layout usando ReportLab Canvas."""

    if not charts:
        print("Nenhum gráfico para incluir no PDF.")
        return

    width, height = A4
    left = 40
    top = height - 40

    img_width = 500
    img_height = 260
    title_space = 25
    img_space = 20

    c = Canvas(output_file, pagesize=A4)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(left, top, "Relatório de Previsões de Tickets")
    c.setFont("Helvetica", 12)

    y_cursor = top - 60

    for i, item in enumerate(charts):

        titulo = item["titulo"]
        imagem_bytes = item["imagem"]

        if i % 2 == 0 and i != 0:
            c.showPage()
            c.setFont("Helvetica-Bold", 20)
            c.drawString(left, top, "Relatório de Previsões de Tickets")
            c.setFont("Helvetica", 12)
            y_cursor = top - 60

        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, y_cursor, f"📈 {titulo}")
        y_cursor -= title_space
        img = ImageReader(BytesIO(imagem_bytes))
        c.drawImage(
            img, left, y_cursor - img_height, width=img_width, height=img_height
        )

        y_cursor -= img_height + img_space

    c.save()
    print(f"📄 PDF gerado com sucesso: {output_file}")


def generate_metrics_pdf(charts, output_file="relatorio_metrics.pdf"):
    """Gera um PDF com layout similar ao relatório de previsões."""

    if not charts:
        print("Nenhum gráfico disponível para gerar o PDF.")
        return

    width, height = A4

    left = 40
    top = height - 40
    img_width = 500
    img_height = 260
    title_space = 25
    img_space = 20

    c = Canvas(output_file, pagesize=A4)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(left, top, "Relatório de Métricas de Tickets")
    c.setFont("Helvetica", 12)

    y_cursor = top - 60

    for i, (title, chart_path) in enumerate(charts):
        if i % 2 == 0 and i != 0:
            c.showPage()
            c.setFont("Helvetica-Bold", 20)
            c.drawString(left, top, "Relatório de Métricas de Tickets")
            c.setFont("Helvetica", 12)
            y_cursor = top - 60
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, y_cursor, f"📊 {title}")

        y_cursor -= title_space
        img = ImageReader(chart_path)
        c.drawImage(
            img, left, y_cursor - img_height, width=img_width, height=img_height
        )

        y_cursor -= img_height + img_space

    c.save()
    print(f"📄 PDF gerado com sucesso: {output_file}")
