-- sbj.subject LIKE 'R:GS.%' OR sbj.subject = 'P:4295911963'
-- sbj.subject LIKE 'R:JPM.%' OR sbj.subject = 'P:5000021791'
-- sbj.subject LIKE 'R:GM.%' OR sbj.subject = 'P:4298546138'
-- sbj.subject LIKE 'R:CVX.%' OR sbj.subject = 'P:4295903744'
-- sbj.subject LIKE 'R:DIS.%' OR sbj.subject = 'P:5064610769'
-- sbj.subject LIKE 'R:PFE.%' OR sbj.subject = 'P:4295904722'
-- sbj.subject LIKE 'R:F.%' OR sbj.subject = 'P:4295903068'
-- sbj.subject LIKE 'R:BA.%' OR sbj.subject = 'P:4295903076'
-- sbj.subject LIKE 'R:WMT.%' OR sbj.subject = 'P:4295905298'
-- sbj.subject LIKE 'R:INTC.%' OR sbj.subject = 'P:4295906830'
-- sbj.subject LIKE 'R:AMC.%' OR sbj.subject = 'P:5000074163'
-- sbj.subject LIKE 'R:MRNA.%' OR sbj.subject = 'P:5066596648'
-- sbj.subject LIKE 'R:CRM.%' OR sbj.subject = 'P:4295915633'
-- sbj.subject LIKE 'R:QCOM.%' OR sbj.subject = 'P:4295907706'
-- sbj.subject LIKE 'R:PYPL.%' OR sbj.subject = 'P:5045870012'
-- sbj.subject LIKE 'R:JNJ.%' OR sbj.subject = 'P:4295904341'
-- sbj.subject LIKE 'R:NKE.%' OR sbj.subject = 'P:4295904620'
-- sbj.subject LIKE 'R:BAC.%' OR sbj.subject = 'P:8589934339'
-- sbj.subject LIKE 'R:MRK.%' OR sbj.subject = 'P:4295904886'
-- sbj.subject LIKE 'R:XOM.%' OR sbj.subject = 'P:4295912121'
-- sbj.subject LIKE 'R:LLY.%' OR sbj.subject = 'P:4295904414'
-- sbj.subject LIKE 'R:TGT.%' OR sbj.subject = 'P:4295912282'
-- sbj.subject LIKE 'R:C.%' OR sbj.subject = 'P:8589934316'
-- sbj.subject LIKE 'R:CMCSA.%' OR sbj.subject = 'P:4295908573'
-- sbj.subject LIKE 'R:SBUX.%' OR sbj.subject = 'P:4295908005'
-- sbj.subject LIKE 'R:MCD.%' OR sbj.subject = 'P:4295904499'
-- sbj.subject LIKE 'R:MS.%' OR sbj.subject = 'P:4295904557'
-- sbj.subject LIKE 'R:FDX.%' OR sbj.subject = 'P:4295912126'
-- sbj.subject LIKE 'R:ZM.%' OR sbj.subject = 'P:5038054958'
-- sbj.subject LIKE 'R:WFC.%' OR sbj.subject = 'P:8589934175'
-- sbj.subject LIKE 'R:ADBE.%' OR sbj.subject = 'P:4295905431'
-- sbj.subject LIKE 'R:VZ.%' OR sbj.subject = 'P:4295911976'
-- sbj.subject LIKE 'R:V.%' OR sbj.subject = 'P:4298015179'
-- sbj.subject LIKE 'R:ORCL.%' OR sbj.subject = 'P:4295907485'
-- sbj.subject LIKE 'R:GE.%' OR sbj.subject = 'P:4295903128'
-- sbj.subject LIKE 'R:T.%' OR sbj.subject = 'P:4295904853'
-- sbj.subject LIKE 'R:CVS.%' OR sbj.subject = 'P:4295903627'
-- sbj.subject LIKE 'R:COST.%' OR sbj.subject = 'P:4295912987'
-- sbj.subject LIKE 'R:AVGO.%' OR sbj.subject = 'P:5060689053'
-- sbj.subject LIKE 'R:HD.%' OR sbj.subject = 'P:4295903148'
-- sbj.subject LIKE 'R:CSCO.%' OR sbj.subject = 'P:8590925492'
-- sbj.subject LIKE 'R:BMY.%' OR sbj.subject = 'P:4295903619'
-- sbj.subject LIKE 'R:AMGN.%' OR sbj.subject = 'P:4295905537'
-- sbj.subject LIKE 'R:LOW.%' OR sbj.subject = 'P:4295904432'
-- sbj.subject LIKE 'R:TMUS.%' OR sbj.subject = 'P:4295900188'
-- sbj.subject LIKE 'R:AXP.%' OR sbj.subject = 'P:55838320007'
-- sbj.subject LIKE 'R:COP.%' OR sbj.subject = 'P:4295903051'
-- sbj.subject LIKE 'R:PEP.%' OR sbj.subject = 'P:4295904718'
-- sbj.subject LIKE 'R:DE.%' OR sbj.subject = 'P:4295903104'
-- sbj.subject LIKE 'R:IBM.%' OR sbj.subject = 'P:4295904307'
-- sbj.subject LIKE 'R:CAT.%' OR sbj.subject = 'P:4295903678'
-- sbj.subject LIKE 'R:NVDA.%' OR sbj.subject = 'P:4295914405'
-- sbj.subject LIKE 'R:AMD.%' OR sbj.subject = 'P:4295903297'
-- sbj.subject LIKE 'R:AAPL.%' OR sbj.subject = 'P:4295905573'
-- sbj.subject LIKE 'R:TSLA.%' OR sbj.subject = 'P:4297089638'
-- sbj.subject LIKE 'R:AMZN.%' OR sbj.subject = 'P:4295905494'
-- sbj.subject LIKE 'R:MSFT.%' OR sbj.subject = 'P:4295907168'
-- sbj.subject LIKE 'R:NFLX.%' OR sbj.subject = 'P:4295902158'
-- sbj.subject LIKE 'R:COIN.%' OR sbj.subject = 'P:5037999423'
-- sbj.subject LIKE 'R:GOOG.%' OR sbj.subject = 'P:5030853586'
-- sbj.subject LIKE 'R:BABA.%' OR sbj.subject = 'P:5000066483'
-- sbj.subject LIKE 'R:NIO.%' OR sbj.subject = 'P:5064707091'


WITH RankedNews AS (
    SELECT
        news.item_id,
        news.first_created AS dates,
        sbj.subject,
        news.headline,
        REGEXP_REPLACE(news.headline, '^\d+', '') AS normalized_headline, -- To ignore the difference of Update 1,2,3,4,5...
        news.body,
        ROW_NUMBER() OVER (PARTITION BY REGEXP_REPLACE(news.headline, '^\d+', '') ORDER BY news.first_created DESC) as rn
    FROM public.item_data news
    INNER JOIN public.data_subject sbj ON news.item_id = sbj.item_id
    WHERE
        news.first_created BETWEEN '2021-08-01 00:00:00 America/New_York' AND '2023-05-30 23:59:59 America/New_York'
        AND news.item_language = 'en'
        AND (
            sbj.subject LIKE 'R:CAT.%' OR sbj.subject = 'P:4295903678'
        )
)

SELECT
    item_id,
    dates,
    subject AS symbols,
    headline,
    body
FROM RankedNews
WHERE rn = 1
AND body IS NOT NULL
AND LENGTH(body) > 300
ORDER BY dates ASC;
