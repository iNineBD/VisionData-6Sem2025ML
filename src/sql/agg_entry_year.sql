-- agrupamento ano
select
d.Year,
sum(t.QtTickets) as TotalTickets
from dbo.Fact_Tickets t
join dbo.Dim_Dates d on t.EntryDateKey = d.DateKey
group by
d.Year
order by
d.Year
