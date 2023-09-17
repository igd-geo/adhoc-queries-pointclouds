# TODOs für Diss

- [ ] Support für Importance Query
    - Challenge: Attribut muss on-the-fly berechnet werden. Fragen:
        - Wie integriere ich das in den Rest der Architektur? 
        - Lohnt es sich die Daten zu cachen? Also dem input_layer sowas wie `get_additional_data` zu geben. Sowas wie Importance könnte dann in ner eigenen Datei stehen, pro Input-Datei, dann könnte man das relativ einfach laden. Oder irgendne andere Struktur, ist ja eigentlich egal, hauptsache schnelles I/O und schnell zu identifizieren
- [ ] Support für knn query
    - Challenge: Ist standardmäßig mit `ORDER BY` implementiert, das kann mein tool nicht da es komplett asynchron ist. Würde halt auch out-of-core sort erfordern
        - Korrektur: Eventuell kann ich `ORDER BY` doch implementieren, wenn es zusammen mit einem `TAKE N` verwendet wird. Dann kann ich einen min-heap nutzen mit einer vordefinierten Kapazität (e.g. `MinHeap<(Attribute, GlobalPointIndex)>`)
- [x] Support für LAST Dateiformat
- [x] Support für LAZ Dateiformat
- [ ] Support für LAZER Dateiformat
- [ ] Support für 3D Tiles Dateiformat