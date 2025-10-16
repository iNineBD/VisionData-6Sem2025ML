-- agrupamento ano e mês por comanpanhia
SELECT
    t.CompanyKey,
    d.Year,
    d.Month,
    SUM(t.QtTickets) AS TotalTickets
FROM dbo.Fact_Tickets t
JOIN dbo.Dim_Dates d 
    ON t.EntryDateKey = d.DateKey
GROUP BY
    t.CompanyKey,
    d.Year,
    d.Month
ORDER BY
    t.CompanyKey,
    d.Year,
    d.Month;
