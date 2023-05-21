# Ideas for improvements

- Generate an index on the fly
    - The index stores chunks with the min/max values of each attribute in that chunk
    - Chunks could be e.g. 10k points in size
    - While scanning first (without an index) we can create a chunk header for each chunk, but only for the queried attribute(s)
        - So if we query by bounds, we compute an AABB for each chunk. If we query by object class, we create a class histogram, and so on...
    - Upon further scans, we first consult the index to find the matching chunks
    - Within each chunk, we can potentially refine the chunk, e.g. by calculating first half and second half chunks, and compare them if they are sufficiently different
    - For uncompressed data, we could also restructure the data (i.e. sort the points), but this only works for a single attribute
- The search should work stream-based so that we can stream the results into whatever application we want
    - In particular relevant for rendering / sending the data to a remote application

## TODOs

- Refinement does not work currently. I started something but then got stuck in too many optimizations. Would have to go back to the concept phase
- IF I really want to implement this, maybe start with a really small, unoptimized POC and go from there? e.g. only index positions and only down to LAZ-block granularity. Demonstrating a progressive index for LAZ blocks itself could already be a good result, as this could counteract the problems that the regular ad-hoc query implementation has with compressed files

## Session 14.05.23

- Habe ein ganz allgemeines Index-Refinement für WITHIN Position3D implementiert, die Refinements werden erstellt, allerdings noch nicht in den Index übernommen (da fehlt nur noch eine Implementierung von `BlockIndex::apply_refinements`)
- Habe runtime logging implementiert, um zu gucken wie sich die Laufzeit für die Query-Evaluierung, Index refinement, und Datenextraktion unterscheiden
    - Datenextraktion ist Faktor 5-10 intensiver als Query Evaluierung, und Faktor 3-5 intensiver als Refinement. Refinement ca. 2x so intensiv wie Evaluation
- Habe dann schon angefangen darüber nachzudenken, wie man Extraktion effizienter machen kann. Profiler sagt, dass es überwiegend `memmove` instructions sind, macht auch Sinn, ist eigentlich nur Daten von Buffer zu Buffer kopieren. Ideen:
    - Statt in einen `InterleavedPointBuffer` zu gehen könnte man direkt LAS point records extrahieren
    - Der Extractor könnte einfach `Vec<Range<usize>>` zurückliefern, und die Zielanwendung könnte dann die Konvertierung übernehmen. Könnte effizienter sein wenn ich z.B. direkt 3D Tiles extrahieren will oder sowas, dann muss ich nicht den Umweg über den pasture Buffer machen
- Ganz interessante Ergebnisse erstmal, TODOs für nächste Session wären:
    - Index refinement einbauen
    - Testszenarien aufbauen, die Index Refinement demonstrieren, also z.B. gleiche Query mehrmals, oder unterschiedliche Queries die aber evtl auch davon profitieren
        - Cool wäre, eine realistische Query zu haben aus einer 3D Anwendung raus, also z.B. Fibre3D
    - Index persistieren auf der Platte
    - Wiss. Frage: Wenn jetzt schon klar ist, dass data extraction den größten Teil der Zeit in Anspruch nimmt, in wieweit ergibt es dann noch Sinn, an Index/Query rumzubasteln? Sollte man nicht erstmal die Extraktion beschleunigen? 
    - Queries für LAZ ebenfalls implementieren, hier könnten die Ergebnisse anders aussehen und da könnte der Index deutlich wichtiger sein

## 15.05.23

- Vermutung: Ich muss erstmal eine MENGE Test-Szenarien aufbauen, sonst kriege ich die Komplexität von dem tool nicht in den Griff. Performance ist da zweitrangig, wirklich erstmal auflisten und implementieren, was ich alles gerne demonstrieren möchte
    - AABB queries haben wir ja schon, die sind einigermaßen trivial
    - LOD queries
        - Wie sieht die zugehörige QueryExpression aus? LOD matcht vermutlich so gut wie alle Blocks, da hilft der Index wenig. Man bräuchte also eher einen 'Importance-Index' (z.B. mit reverse-morton sorted Dateien)
        - First step: LOD abbilden über den Result-Collector / Extractor. Z.B. als `LIMIT(n)` um die ersten `n` Punkte zu kriegen, oder `IMPORTANCE(percentage)` für die ersten `percentage` Prozent der wichtigsten Punkte
    - Classification queries
    - Kombinierte Queries
    - Anwendungsfälle??
        - Direkte Visualisierung wäre toll, eventuell mit Tobias' Renderer oder direkt Potree/Cesium?
- Mal mit Michel und Tobias reden über deren progressive indexing ideen, vllt kann man hier noch was kombinieren? Geht aber letztlich ja nur drum für meine Diss noch paar Resultate zu kriegen
- Mit Kevin nochmal diskutieren was man aufbauend auf seinem Cloud-based tool noch machen könnte
    - Könnte dieses Tool hier auch was für die Cloud sein? 

## Ablauf query

- parsen einer query in eine `QueryExpression` (quasi ein AST)
- query auf dataset anwenden, um alle potentiellen Stellen in allen Dateien zu identifizieren, die Punkte enthalten die der query entsprechen
    - query rekursiv anwenden, beginnend bei den Blatt-Knoten (atomare `QueryExpression`)
    - Beispiele:
        - `within`: Braucht exakt einen `BlockIndex`, also alle Blöcke dieses `BlockIndex` mit dem range von `within` verschneiden
        - `equals`: Braucht exakt einen `BlockIndex`, alle Blöcke dieses `BlockIndex` auf Gleichheit prüfen
    - Das Resultat jeder atomaren `QueryExpression` ist ein 'hit rating', d.h. wie gut dieser Block auf die Query passt
        - Aktuell nur `MatchAll`, `MatchSome`, und `NoMatch`
    - Interne Knoten (`and`, `or`) kombinieren dann die Ergebnisse der Blatt-Knoten
        - Unklar, wie genau das Kombinieren aussieht. Resultat sind wieder eine Reihe an Blöcken, diese können aber z.B. größer sein, als die einzelnen Blöcke der Indices
            - **Erklärung** Index A hat 10 Blöcke, Index B hat nur einen Block. Wir haben eine `or` query, von Index A matcht nichts, aber der eine Block von Index B. Der ist natürlich viel größer als alle Blöcke in Index A
            - **Warum relevant?** Für das Index-Refinement: Das Resultat einer query über Indices kann nicht direkt angewendet werden, um Blöcke zu refinen, da wir nicht mehr genau wissen, zu welchen Blöcken das query Resultat korrespondiert
            - **Mögliche Lösung** Mit dem Query-Resultat auch eine Liste an Blöcken mitgeben, die potentiell verfeinert werden könnten
                - Optimierung: Hier nur die Blöcke verfeinern, die `MatchSome` als Resultat gegeben haben (da das die Blöcke sind, die unklar sind. Wären sie feiner gewesen hätte man ja vielleicht `MatchAll` oder `None` rausgekriegt, was immer besser ist)
- mit dem Resultat der groben Query (eine Menge an Blöcken) die Fein-Query ausführen
    - alle Blöcke die `MatchAll` haben können bereits vollständig extrahiert werden
    - für `MatchSome` Blöcke muss der Block eingelesen werden und jeder Punkt mit der Query geprüft werden
    - Resultat der Fein-Query könnte eins von 2 Dingen sein:
        - Eine Menge an Blöcken bei denen ALLE Blöcke `MatchAll` haben
        - Die tatsächlichen Punkte, aus den Dateien extrahiert
        - **TODO** Vorteile/Nachteile evaluieren
- Falls Fein-Query Blöcke zurückliefert, können diese nun extrahiert bzw. an andere Anwendungen weitergereicht werden
    - Extraktion kostet ja die meiste Zeit wie ich festgestellt habe...

## Ablauf Refinement

- Um die Indices zu verfeinern braucht es eine Liste von Blöcken, die verfeinert werden *könnten*
    - Das wären alle Blöcke die in der aktuellen Grob-Query `MatchSome` zurückgeliefert haben
- Wie viele / welche Blöcke tatsächlich verfeinert werden kann über irgendeinen Algorithmus bestimmt werden
    - z.B. nach 'Wichtigkeit', verfügbare Zeit für Refinement berücksichtigen etc.
- Verfeinern selbst ist relativ einfach:
    - Entscheiden, wie verfeinert werden soll
        - Einen Block fix in `N` Blöcke aufsplitten (z.B. jeder neue Block mit 1/N Punkten)
        - Intelligent splitten basierend auf irgendwelchen Metriken (z.B. AABB der ersten `k` Punkte berechnen, so lange bis die AABB für `k+1` Punkte einen Threshold überschreitet)
        - Für unkomprimierte fixed-width Dateien wie LAS könnte man auch einfach sortieren (dann hätte man aber keinen read-only Index mehr)
            - Wäre auch möglich, Dateien zu kopieren und in optimierter Form abzulegen. Dann würde der Block auf die kopierte Datei zeigen statt auf die Originaldatei
    - Alle Punkte im Block einlesen und die Verfeinerung anwenden

### Anmerkungen

- Ich dachte ursprünglich, dass ich die Fein-Query und das Refinement kombinieren kann, um mir zu sparen die Datei mehrmals zu lesen (da das Parsen aufwändig sein könnte). Ist das wirklich notwendig? Bzw. wie groß ist der Performance-Gewinn dadurch? Da ich ja nur für ein einziges Attribut verfeinere ist das Lesen nicht so aufwändig wie die finale Extraktion

## Allgemeine Ideen

- Ich möchte einen SQL statement parser für meine Queries, das wäre so viel einfacher
    - `SELECT position,classification FROM dataset_name WHERE position within AABB(123, 456, 789) and classification = 4 LIMIT importance(1%)`
- Anbindung an eine Visualisierungs-Anwendung wäre mega praktisch
    - Query abschicken und direkt das Ergebnis in 3D visualisiert kriegen
    - Noch geiler wäre, wenn die Visualisierung direkt queries schicken würde während man sich bewegt
        - Man bräuchte eine Art Basis-Struktur um das machen zu können, also zumindest einen Octree der sagt 'diese Zellen gibt es' damit die Visualisierung dann die entsprechenden AABB-Queries (mit importance) schicken kann
        - **Frage** Wie schnell könnte ich einen einfachen Octree für einen Datensatz erstellen? Das wäre ja ein ähnlicher Algorithmus wie der counting sort vom Markus, aber ohne sorting. Das sollte doch eigentlich sehr schnell gehen denke ich? Aber geht vermutlich nicht 100% als Online-Algorithmus da ich ja irgendwann Nodes entweder splitten oder zusammenführen muss, und dafür wissen muss welche Punkte in den Nodes liegen. Aber man könnte eine fixe Tiefe des Octrees angeben und den erstellen. Ist dann letztlich eher ein Grid, aber wäre ja auch ok, geht ja nur drum die Struktur zu kennen damit man dann queries abfeuern kann. Und Zellen mit zu wenigen Punkten zusammenfassen geht auch ohne dass ich weiß, welche Punkte genau drin liegen, nur splitten geht halt nicht. 
- Ich muss endlich mal klären wie das mit der Performance ist wenn man einzelne Punkte liest vs. wenn man einfach einen ganzen Block Punkte liest und dann die rausschmeißt die einen nicht interessieren...
    - Also `read_range(a..b).filter(index_matches)` vs. `index_matches.map(|index| read_at_index(index))`