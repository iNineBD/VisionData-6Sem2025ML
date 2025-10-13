-- agrupamento ano e mês
select
d.Year,
d.Month,
sum(t.QtTickets) as TotalTickets
from dbo.Fact_Tickets t
join dbo.Dim_Dates d on t.EntryDateKey = d.DateKey
group by
d.Year, d.Month
order by
d.Year, d.Month
