<?xml version="1.0" encoding="UTF-8"?>
<xsl:transform version="1.1" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns="http://www.w3.org/1999/xhtml" 
  xmlns:tei="http://www.tei-c.org/ns/1.0" 
  exclude-result-prefixes="tei"
  xmlns:exsl="http://exslt.org/common"
  extension-element-prefixes="exsl"
  >
  <xsl:include href="../../../teinte-xsl/tei_txt/tei_markdown.xsl"/>
  <xsl:param name="filename"/>
  <xsl:param name="outdir"/>
  <xsl:variable name="issued">
    <xsl:variable name="date" select="/tei:TEI/tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date"/>
    <xsl:choose>
      <xsl:when test="$date/@when">
        <xsl:value-of select="$date/@when"/>
      </xsl:when>
      <xsl:when test="$date != ''">
        <xsl:value-of select="$date"/>
      </xsl:when>
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="chapcount" select="count(/tei:TEI/tei:text//tei:div[@type='chapter'])"/>
  <xsl:variable name="artcount" select="count(/tei:TEI/tei:text//tei:div[@type='article'])"/>
  <xsl:variable name="chapformat">
    <xsl:choose>
      <xsl:when test="$chapcount &gt; 99">001</xsl:when>
      <xsl:otherwise>01</xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
  <xsl:variable name="yamline">
    <xsl:text>---</xsl:text>
    <xsl:value-of select="$lf"/>
  </xsl:variable>
  <xsl:variable name="bookid">
    <xsl:choose>
      <xsl:when test="/*/@xml:id">
        <xsl:value-of select="/*/@xml:id"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="substring-before($filename, '_')"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>
  <xsl:template match="/">
    <xsl:value-of select="$lf"/>
    <xsl:choose>
      <xsl:when test="contains('
        ddr19290300mip
        ddr19341025polpers
        ddr19361100pm
        ddr19390600ao
        ddr19421200partdia
        ddr19441100partdia
        ddr19460615lba1
        ddr19820900paysan
        ddr19680314jde
        ddr19560200ao
        ',$bookid)"/>
      <xsl:when test="$chapcount > 1">
        <xsl:apply-templates select="/tei:TEI/tei:text//tei:div[@type='chapter']"/>
      </xsl:when>
      <xsl:when test="$artcount > 1">
        <xsl:apply-templates select="/tei:TEI/tei:text//tei:div[@type='article']"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:variable name="href" select="concat($outdir, $bookid, '.md')"/>
        <xsl:value-of select="$href"/>
        <xsl:value-of select="$lf"/>
        <exsl:document href="{$href}" method="text" omit-xml-declaration="yes" encoding="UTF-8" indent="yes">
          <xsl:value-of select="$yamline"/>
          <xsl:text>identifier: </xsl:text>
          <xsl:value-of select="$bookid"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>creator: </xsl:text>
          <xsl:value-of select="$author1"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>created: </xsl:text>
          <xsl:value-of select="$docdate"/>
          <xsl:value-of select="$lf"/>
          <xsl:if test="$issued != '' and substring($issued, 1, 4) != substring($docdate, 1, 4)">
            <xsl:text>issued: </xsl:text>
            <xsl:value-of select="$issued"/>
            <xsl:value-of select="$lf"/>
          </xsl:if>
          <xsl:text>title: </xsl:text>
          <xsl:call-template name="quote">
            <xsl:with-param name="value" select="$doctitle"/>
          </xsl:call-template>
          <xsl:value-of select="$lf"/>
          <xsl:value-of select="$yamline"/>
          <xsl:value-of select="$lf"/>
          <xsl:apply-templates select="/tei:TEI/tei:text/tei:body/node()" mode="md"/>
        </exsl:document>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>


  <xsl:template match="tei:div">
    <xsl:variable name="no">
      <xsl:number count="tei:div[@type = 'chapter']" format="{$chapformat}" from="tei:text" level="any"/>
    </xsl:variable>
    <xsl:variable name="docid">
      <xsl:choose>
        <xsl:when test="@xml:id">
          <xsl:value-of select="@xml:id"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$bookid"/>
          <xsl:text>-</xsl:text>
          <xsl:value-of select="$no"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:variable name="href" select="concat($outdir, $docid, '.md')"/>
    <xsl:variable name="title">
      <xsl:variable name="raw">
        <xsl:apply-templates select="tei:head" mode="title"/>
      </xsl:variable>
      <xsl:value-of select="normalize-space($raw)"/>
    </xsl:variable>
    <xsl:value-of select="$href"/>
    <xsl:value-of select="$lf"/>
    <exsl:document href="{$href}" method="text" omit-xml-declaration="yes" encoding="UTF-8" indent="yes">
      <xsl:value-of select="$yamline"/>
      <xsl:text>identifier: </xsl:text>
      <xsl:value-of select="$docid"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>creator: </xsl:text>
      <xsl:value-of select="$author1"/>
      <xsl:value-of select="$lf"/>
      <xsl:choose>
        <xsl:when test="@when">
          <xsl:text>created: </xsl:text>
          <xsl:value-of select="@when"/>
          <xsl:value-of select="$lf"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text>created: </xsl:text>
          <xsl:value-of select="$docdate"/>
          <xsl:value-of select="$lf"/>
          <xsl:if test="$issued != '' and substring($issued, 1, 4) != substring($docdate, 1, 4)">
            <xsl:text>issued: </xsl:text>
            <xsl:value-of select="$issued"/>
            <xsl:value-of select="$lf"/>
          </xsl:if>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:text>isPartOf: </xsl:text>
      <xsl:call-template name="quote">
        <xsl:with-param name="value" select="$doctitle"/>
      </xsl:call-template>
      <xsl:value-of select="$lf"/>
      <xsl:text>title: </xsl:text>
      <xsl:call-template name="quote">
        <xsl:with-param name="value" select="$title"/>
      </xsl:call-template>
      <xsl:value-of select="$lf"/>
      <xsl:value-of select="$yamline"/>
      <xsl:value-of select="$lf"/>
      <xsl:choose>
        <xsl:when test="tei:div[@type='chapter']">
          <xsl:variable name="first-div" select="(
              tei:back
            | tei:body
            | tei:front
            | tei:group
            | tei:div
            | tei:div0
            | tei:div1
            | tei:div2
            | tei:div3
            | tei:div4
            | tei:div5
            | tei:div6
            )[1]"/>
          <xsl:apply-templates select="$first-div/preceding-sibling::node()" mode="md"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates mode="md"/>
        </xsl:otherwise>
      </xsl:choose>
    </exsl:document>
  </xsl:template>
  <xsl:template match="tei:note" mode="md"/>
  <xsl:template match="tei:hi" mode="md">
    <xsl:apply-templates/>
  </xsl:template>
  <xsl:template name="quote">
    <xsl:param name="value"/>
    <xsl:choose>
      <xsl:when test="substring($value, 1, 1) = '[' or contains($value, ':')">
        <xsl:text>"</xsl:text>
        <xsl:value-of select="translate($value, '&quot;', ' ')"/>
        <xsl:text>"</xsl:text>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$value"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <xsl:template match="*[@type='noindex']" mode="md"/>
</xsl:transform>
