{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text.IO as T
import qualified Data.Text as Text
import qualified Data.ByteString.Lazy.Char8 as BL8

import System.Environment (getArgs)
import Data.Char (isSpace)
import Data.List (isPrefixOf, sort, nub, intercalate)
import Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Set as Set
import qualified Data.Map.Strict as Map

import Control.Monad (forM_)
import Control.Monad.State.Strict (State, runState, modify', get)

import Data.Aeson (ToJSON(..), (.=), object, encode)
import Data.Foldable (foldlM)

import Cryptol.Parser (parseModule, defaultConfig, Config(..))
import Cryptol.Parser.AST
  ( PName, TopDecl(..), Decl(..), TopLevel(..)
  , Bind(..), BindImpl(..), Schema(..), TySyn(..), Type(..), Prop(..)
  , TParam(..) 
  , Expr(..), Match(..), CaseAlt(..), UpdField(..), PropGuardCase(..), Pattern(..)
  , mDecls, tsName, bindImpl, bindParams, value
  )

import Cryptol.Parser.Position (thing)
import Cryptol.Utils.PP (pretty)
import Cryptol.Utils.RecordMap (recordElements, displayFields)

--------------------------------------------------------------------------------
-- JSON types
--------------------------------------------------------------------------------

data Ref
  = RefDef String
  | RefLocal String
  | RefVar String
  | RefPrim String
  | RefImport String
  | RefImportMember String String
  | RefOp String
  deriving (Eq, Ord)

instance ToJSON Ref where
  toJSON r =
    case r of
      RefDef s                -> object ["def"    .= s]
      RefLocal s              -> object ["local"  .= s]
      RefVar s                -> object ["var"    .= s]
      RefPrim s               -> object ["prim"   .= s]
      RefImport s             -> object ["import" .= s]
      RefImportMember imp mem -> object ["import" .= imp, "member" .= mem]
      RefOp s                 -> object ["op"     .= s]

data MCCNode = MCCNode
  { nodeId  :: Int
  , nodeKind :: String
  , nodeLabel :: String
  , nodeDetails :: [String]
  }

instance ToJSON MCCNode where
  toJSON n =
    object
      [ "id"      .= nodeId n
      , "kind"    .= nodeKind n
      , "label"   .= nodeLabel n
      , "details" .= nodeDetails n
      ]

data MCCEdge = MCCEdge
  { edgeFrom  :: Int
  , edgeTo    :: Int
  , edgeKind  :: String
  , edgeLabel :: Maybe String
  }

instance ToJSON MCCEdge where
  toJSON e =
    object
      [ "from"  .= edgeFrom e
      , "to"    .= edgeTo e
      , "kind"  .= edgeKind e
      , "label" .= edgeLabel e
      ]

data MCCGraph = MCCGraph
  { mccEntry :: Int
  , mccExit  :: Int
  , mccNodes :: [MCCNode]
  , mccEdges :: [MCCEdge]
  }

instance ToJSON MCCGraph where
  toJSON g =
    object
      [ "entry" .= mccEntry g
      , "exit"  .= mccExit g
      , "nodes" .= mccNodes g
      , "edges" .= mccEdges g
      ]

data DefNode = DefNode
  { name       :: String
  , kind       :: String
  , signature  :: Maybe String
  , params     :: [String]
  , locals     :: [String]
  , references :: [Ref]
  , mcc        :: Maybe MCCGraph
  }

instance ToJSON DefNode where
  toJSON d =
    object
      [ "name"       .= name d
      , "kind"       .= kind d
      , "signature"  .= signature d
      , "params"     .= params d
      , "locals"     .= locals d
      , "references" .= references d
      , "mcc"        .= mcc d
      ]

data Graph = Graph
  { imports     :: [String]
  , definitions :: [DefNode]
  }

instance ToJSON Graph where
  toJSON g =
    object
      [ "imports"     .= imports g
      , "definitions" .= definitions g
      ]

--------------------------------------------------------------------------------
-- CLI
--   toy-cryptol-graph <file.cry> [prims.yaml]
--------------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  case args of
    [fp]        -> run fp Nothing
    [fp, yml]   -> run fp (Just yml)
    _           -> putStrLn "Usage: toy-cryptol-graph <file.cry> [cryptol_primitives.yaml]"

run :: FilePath -> Maybe FilePath -> IO ()
run fp mYaml = do
  src <- T.readFile fp
  let cfg = defaultConfig { cfgSource = fp }

  primSet <- case mYaml of
    Nothing  -> pure Set.empty
    Just yfp -> loadPrimitiveYaml yfp

  case parseModule cfg src of
    Left err -> do
      putStrLn "Parse error:"
      print err
    Right mods -> do
      let decls      = concatMap mDecls mods
          importStrs = importLinesToStrings (Text.unpack src)
          aliasMap   = buildImportAliasMap importStrs

          sigMap     = buildSignatureMap decls

          topEntries = topLevelEntries decls
          topNames   = Set.fromList [ pretty (teName e) | e <- topEntries ]

          defNodes =
            concatMap (defNodesForEntry primSet aliasMap topNames sigMap) topEntries

          g = Graph { imports = importStrs, definitions = defNodes }

      BL8.putStrLn (encode g)

--------------------------------------------------------------------------------
-- Primitive/Symbol YAML (very simple YAML-list extractor)
-- Accepts YAML like:
--   primitives:
--     - split
--     - join
--   symbols:
--     - "(@)"
--     - "(>>>)"
--------------------------------------------------------------------------------

loadPrimitiveYaml :: FilePath -> IO (Set.Set String)
loadPrimitiveYaml yfp = do
  s <- readFile yfp
  let items = mapMaybe parseListItem (lines s)
  pure (Set.fromList items)
  where
    parseListItem :: String -> Maybe String
    parseListItem line0 =
      let line1 = takeWhile (/= '#') line0
          t     = trim line1
      in case t of
           ('-':rest) ->
             let v = stripQuotes (trim rest)
             in if null v then Nothing else Just v
           _ -> Nothing

stripQuotes :: String -> String
stripQuotes s =
  case s of
    ('"':xs) -> dropWhileEnd (== '"') xs
    ('\'':xs)-> dropWhileEnd (== '\'') xs
    _        -> s

--------------------------------------------------------------------------------
-- Option 1: emit separate DefNodes for where-bound locals
--------------------------------------------------------------------------------

defNodesForEntry
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Map.Map String (Schema PName)
  -> TopEntry
  -> [DefNode]
defNodesForEntry primSet aliasMap topNames sigMap e =
  let defNameStr = pretty (teName e)
      defKindStr = topKindToText (teKind e)
      sigStr     = fmap pretty (Map.lookup defNameStr sigMap)

      paramSet   = teParams e
      localSet   = teLocals e

      rawRefs =
        teRawRefs e `Set.union`
        maybe Set.empty namesInSchema (Map.lookup defNameStr sigMap)

      refs =
        sort . nub $
          classifyRefs primSet aliasMap topNames paramSet localSet rawRefs

      mccGraph =
        case teImpl e of
          Nothing         -> Nothing
          Just (VExpr ex) -> Just (buildMCCForExpr primSet aliasMap topNames paramSet localSet defNameStr ex)
          Just (VGuards cs) ->
            Just (buildMCCForGuards primSet aliasMap topNames paramSet localSet defNameStr cs)

      topNode =
        DefNode
          { name       = defNameStr
          , kind       = defKindStr
          , signature  = sigStr
          , params     = sort (Set.toList paramSet)
          , locals     = sort (Set.toList localSet)
          , references = refs
          , mcc        = mccGraph
          }

      -- Option 1: add separate DefNodes for every where-bound binding.
      localNodes =
        case teImpl e of
          Nothing         -> []
          Just (VExpr ex) ->
            collectLocalDefNodes primSet aliasMap topNames defNameStr paramSet Set.empty ex
          Just (VGuards cs) ->
            concatMap
              (\c -> collectLocalDefNodes primSet aliasMap topNames defNameStr paramSet Set.empty (pgcExpr c))
              cs

  in topNode : localNodes


-- Walk expressions and pull out EWhere-bound Decl nodes as separate DefNodes.
-- parentQName is the *containing definition name*, used to qualify locals as Parent::local.
collectLocalDefNodes
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> String            -- parent qualified name (e.g. "InvMixColumns")
  -> Set.Set String    -- vars in scope (enclosing params; used to classify captured vars as RefVar)
  -> Set.Set String    -- locals in scope (enclosing where-bound names)
  -> Expr PName
  -> [DefNode]
collectLocalDefNodes primSet aliasMap topDefs parentQName inScopeVars inScopeLocals expr =
  case expr of
    EWhere body ds ->
      let sigLocal  = buildSignatureMapDecls ds
          hereNames = Set.unions (map declBinds ds)        -- names declared by this where
          envLocals = inScopeLocals `Set.union` hereNames  -- locals visible in where scope

          nodesHere =
            concatMap
              (defNodesFromLocalDecl primSet aliasMap topDefs parentQName inScopeVars envLocals sigLocal)
              ds

          -- continue walking the body with the where-locals in scope
          nodesInBody =
            collectLocalDefNodes primSet aliasMap topDefs parentQName inScopeVars envLocals body

      in nodesHere ++ nodesInBody

    -- propagate traversal
    EIf a b c ->
      go inScopeVars inScopeLocals a ++
      go inScopeVars inScopeLocals b ++
      go inScopeVars inScopeLocals c

    ECase scrut alts ->
      go inScopeVars inScopeLocals scrut ++
      concat [ go inScopeVars inScopeLocals rhs | CaseAlt _ rhs <- alts ]

    -- If a where appears inside a comprehension body, binders are in scope there.
    EComp body mss ->
      let ms     = concat mss
          bound  = Set.unions (map matchBinds ms)
          vars'  = inScopeVars `Set.union` bound

          inMatches = concatMap (localsInMatch inScopeVars inScopeLocals) ms
          inBody    = go vars' inScopeLocals body

      in inMatches ++ inBody

    EApp a b      -> go inScopeVars inScopeLocals a ++ go inScopeVars inScopeLocals b
    EInfix a _ _ b-> go inScopeVars inScopeLocals a ++ go inScopeVars inScopeLocals b
    ETuple es     -> concatMap (go inScopeVars inScopeLocals) es
    EList es      -> concatMap (go inScopeVars inScopeLocals) es
    ERecord rec   -> concatMap (go inScopeVars inScopeLocals) [ e | (_,e) <- recordElements rec ]
    ESel e _      -> go inScopeVars inScopeLocals e
    EGenerate e   -> go inScopeVars inScopeLocals e
    EAppT e _     -> go inScopeVars inScopeLocals e
    ETyped e _    -> go inScopeVars inScopeLocals e
    ELocated e _  -> go inScopeVars inScopeLocals e
    EParens e     -> go inScopeVars inScopeLocals e
    ESplit e      -> go inScopeVars inScopeLocals e
    EPrefix _ e   -> go inScopeVars inScopeLocals e
    EUpd me fields ->
      maybe [] (go inScopeVars inScopeLocals) me ++
      concat [ go inScopeVars inScopeLocals e | UpdField _ _ e <- fields ]
    EInfFrom e me ->
      go inScopeVars inScopeLocals e ++
      maybe [] (go inScopeVars inScopeLocals) me

    -- treat other forms as leaves for local-def extraction
    _ -> []
  where
    go vars locs = collectLocalDefNodes primSet aliasMap topDefs parentQName vars locs

    localsInMatch vars locs m =
      case m of
        Match _ e -> go vars locs e
        MatchLet b ->
          let bodyExprs = implBodyExprs (bindImpl b)
          in concatMap (go vars locs) bodyExprs


-- Turn Decl nodes inside a where into separate DefNodes (Parent::localName),
-- and recurse to find nested where-bindings inside each local body.
defNodesFromLocalDecl
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> String                 -- parentQName
  -> Set.Set String         -- vars in scope from outer defs
  -> Set.Set String         -- locals in scope for this where block
  -> Map.Map String (Schema PName)  -- local signature map (unqualified)
  -> Decl PName
  -> [DefNode]
defNodesFromLocalDecl primSet aliasMap topDefs parentQName outerVars envLocals sigLocal decl =
  case dropLocDecl decl of
    DBind b ->
      mkBindNode b

    DRec bs ->
      concatMap mkBindNode bs

    DPatBind p e ->
      case patVarName p of
        Nothing -> []
        Just nm ->
          let localName = pretty nm
              qname     = parentQName ++ "::" ++ localName

              sigStr    = fmap pretty (Map.lookup localName sigLocal)

              localsIn  = localDeclNamesInExpr e
              rawRefs0  = Set.map prettyName (namesInExpr e)
              rawRefs   =
                rawRefs0 `Set.union`
                maybe Set.empty namesInSchema (Map.lookup localName sigLocal)

              varsForClass = outerVars  -- no new params here
              refs =
                sort . nub $
                  classifyRefs primSet aliasMap topDefs varsForClass envLocals rawRefs

              mccGraph =
                Just (buildMCCForExpr primSet aliasMap topDefs varsForClass envLocals qname e)

              node =
                DefNode
                  { name       = qname
                  , kind       = "declaration"
                  , signature  = sigStr
                  , params     = []   -- pattern bind has no params
                  , locals     = sort (Set.toList localsIn)
                  , references = refs
                  , mcc        = mccGraph
                  }

              nested =
                collectLocalDefNodes primSet aliasMap topDefs qname varsForClass envLocals e

          in node : nested

    -- Optional but useful: local type synonyms in where blocks become separate "type" DefNodes.
    DType ts ->
      let localName = pretty (thing (tsName ts))
          qname     = parentQName ++ "::" ++ localName
          ty        = case ts of TySyn _ _ _ t -> t
          rawRefs   = Set.map prettyName (namesInType ty)
          refs =
            sort . nub $
              classifyRefs primSet aliasMap topDefs Set.empty envLocals rawRefs
      in [ DefNode
             { name       = qname
             , kind       = "type"
             , signature  = Nothing
             , params     = []
             , locals     = []
             , references = refs
             , mcc        = Nothing
             }
         ]

    -- signatures handled via buildSignatureMapDecls
    DSignature{} ->
      []

    _ ->
      []
  where
    mkBindNode b =
      let localName = pretty (thing (bName b))
          qname     = parentQName ++ "::" ++ localName

          impl      = bindImpl b
          ownParams = Set.unions (map patBinds (bindParams b))
          bodyExprs = implBodyExprs impl

          sigStr    = fmap pretty (Map.lookup localName sigLocal)

          localsIn  = Set.unions (map localDeclNamesInExpr bodyExprs)
          rawRefs0  = Set.unions (map (Set.map prettyName . namesInExpr) bodyExprs)
          rawRefs   =
            rawRefs0 `Set.union`
            maybe Set.empty namesInSchema (Map.lookup localName sigLocal)

          varsForClass = outerVars `Set.union` ownParams

          refs =
            sort . nub $
              classifyRefs primSet aliasMap topDefs varsForClass envLocals rawRefs

          mccGraph =
            case implToValueImpl impl of
              Nothing         -> Nothing
              Just (VExpr ex) -> Just (buildMCCForExpr primSet aliasMap topDefs varsForClass envLocals qname ex)
              Just (VGuards cs)-> Just (buildMCCForGuards primSet aliasMap topDefs varsForClass envLocals qname cs)

          node =
            DefNode
              { name       = qname
              , kind       = "declaration"
              , signature  = sigStr
              , params     = sort (Set.toList ownParams)  -- own params only
              , locals     = sort (Set.toList localsIn)
              , references = refs
              , mcc        = mccGraph
              }

          nested =
            concatMap
              (collectLocalDefNodes primSet aliasMap topDefs qname varsForClass envLocals)
              bodyExprs

      in node : nested


-- Local (where-scope) signature map.
buildSignatureMapDecls :: [Decl PName] -> Map.Map String (Schema PName)
buildSignatureMapDecls = foldr go Map.empty
  where
    go d acc =
      case d of
        DSignature ns sch ->
          foldr (\ln m -> Map.insert (pretty (thing ln)) sch m) acc ns
        DLocated d' _ ->
          go d' acc
        _ ->
          acc

--------------------------------------------------------------------------------
-- Imports: keep only "Common::GF28 as GF28"
--------------------------------------------------------------------------------

isImportLine :: String -> Bool
isImportLine line =
  case dropWhile isSpace line of
    ('i':'m':'p':'o':'r':'t':' ':_) -> True
    _                               -> False

importLinesToStrings :: String -> [String]
importLinesToStrings srcStr =
  [ dropWhile isSpace (drop (length ("import " :: String)) (dropWhile isSpace line))
  | line <- lines srcStr
  , isImportLine line
  ]

buildImportAliasMap :: [String] -> Map.Map String String
buildImportAliasMap imps =
  Map.fromList (catMaybes (map parseAlias imps))
  where
    parseAlias :: String -> Maybe (String, String)
    parseAlias s =
      case breakOn " as " s of
        Just (modPart, alias) ->
          let alias' = trim alias
              full   = trim modPart ++ " as " ++ alias'
          in if null alias' then Nothing else Just (alias', full)
        Nothing -> Nothing

--------------------------------------------------------------------------------
-- Top-level entry extraction
--------------------------------------------------------------------------------

data TopKind = TKType | TKDecl deriving (Eq)

topKindToText :: TopKind -> String
topKindToText k =
  case k of
    TKType -> "type"
    TKDecl -> "declaration"

data ValueImpl
  = VExpr (Expr PName)
  | VGuards [PropGuardCase PName]

data TopEntry = TopEntry
  { teName   :: PName
  , teKind   :: TopKind
  , teImpl   :: Maybe ValueImpl
  , teParams :: Set.Set String
  , teLocals :: Set.Set String
  , teRawRefs :: Set.Set String
  }

topLevelEntries :: [TopDecl PName] -> [TopEntry]
topLevelEntries = concatMap goTD
  where
    goTD td =
      case td of
        Decl tl -> goDecl (dropLocDecl (tlValue tl))
        _       -> []

    goDecl d =
      case d of
        DType ts@(TySyn _ _ tps _) ->
          let nm      = thing (tsName ts)
              ty      = tySynRHS ts

              -- ✅ type parameters: KeySize, BlockSize, etc.
              tyParams = Set.fromList [ pretty (tpName tp) | tp <- tps ]

              -- ✅ record fields: encrypt, decrypt, etc.
              fields  = recordFieldNamesInType ty

              rawTys  = Set.map prettyName (namesInType ty)
              rawRefs = rawTys `Set.union` fields
          in [ TopEntry nm TKType Nothing tyParams fields rawRefs ]

        DBind b ->
          let nm     = thing (bName b)
              impl   = bindImpl b
              params = Set.unions (map patBinds (bindParams b))
              bodyExprs = implBodyExprs impl
              locals = Set.unions (map localDeclNamesInExpr bodyExprs)
              rawRefs = Set.unions (map (Set.map prettyName . namesInExpr) bodyExprs)
          in [ TopEntry nm TKDecl (implToValueImpl impl) params locals rawRefs ]

        DRec bs ->
          [ let nm     = thing (bName b)
                impl   = bindImpl b
                params = Set.unions (map patBinds (bindParams b))
                bodyExprs = implBodyExprs impl
                locals = Set.unions (map localDeclNamesInExpr bodyExprs)
                rawRefs = Set.unions (map (Set.map prettyName . namesInExpr) bodyExprs)
            in TopEntry nm TKDecl (implToValueImpl impl) params locals rawRefs
          | b <- bs
          ]

        DPatBind p e ->
          case patVarName p of
            Just nm ->
              let locals = localDeclNamesInExpr e
                  rawRefs = Set.map prettyName (namesInExpr e)
              in [ TopEntry nm TKDecl (Just (VExpr e)) Set.empty locals rawRefs ]
            Nothing -> []

        DLocated d' _ -> goDecl d'
        _             -> []

    -- Extract rhs type from TySyn in the safest way with the imports you already have:
    tySynRHS :: TySyn PName -> Type PName
    tySynRHS (TySyn _ _ _ ty) = ty

implToValueImpl :: Maybe (BindImpl PName) -> Maybe ValueImpl
implToValueImpl mi =
  case mi of
    Just (DExpr e)        -> Just (VExpr e)
    Just (DPropGuards cs) -> Just (VGuards cs)
    _                     -> Nothing

implBodyExprs :: Maybe (BindImpl PName) -> [Expr PName]
implBodyExprs mi =
  case mi of
    Just (DExpr e)        -> [e]
    Just (DPropGuards cs) -> [ pgcExpr c | c <- cs ]
    _                     -> []

recordFieldNamesInType :: Type PName -> Set.Set String
recordFieldNamesInType ty =
  case ty of
    -- fs :: Rec (Type PName)  (a RecordMap)
    -- displayFields :: RecordMap a b -> [(a,b)]
    TRecord fs     -> Set.fromList [ pretty f | (f, _) <- displayFields fs ]
    TLocated t _   -> recordFieldNamesInType t
    TParens t _    -> recordFieldNamesInType t
    _              -> Set.empty

patVarName :: Pattern PName -> Maybe PName
patVarName p =
  case dropLocPattern p of
    PVar x -> Just (thing x)
    _      -> Nothing

dropLocPattern :: Pattern PName -> Pattern PName
dropLocPattern (PLocated p _) = dropLocPattern p
dropLocPattern p              = p

patBinds :: Pattern PName -> Set.Set String
patBinds p =
  case dropLocPattern p of
    PVar x        -> Set.singleton (pretty (thing x))     -- Located name
    PWild         -> Set.empty
    PTuple ps     -> Set.unions (map patBinds ps)
    PList ps      -> Set.unions (map patBinds ps)
    PTyped p' _   -> patBinds p'
    PSplit a b    -> patBinds a `Set.union` patBinds b
    PCon _ ps     -> Set.unions (map patBinds ps)
    PRecord rec   -> Set.unions [ patBinds pe | (_,pe) <- recordElements rec ]
    _             -> Set.empty


--------------------------------------------------------------------------------
-- Signature map (explicit signatures only)
--------------------------------------------------------------------------------

buildSignatureMap :: [TopDecl PName] -> Map.Map String (Schema PName)
buildSignatureMap = foldr go Map.empty
  where
    go td acc =
      case td of
        Decl tl ->
          case dropLocDecl (tlValue tl) of
            DSignature ns sch ->
              foldr (\ln m -> Map.insert (pretty (thing ln)) sch m) acc ns
            DLocated d' _ -> go (Decl tl { tlValue = d' }) acc
            _ -> acc
        _ -> acc

--------------------------------------------------------------------------------
-- Ref classification
--------------------------------------------------------------------------------

prettyName :: PName -> String
prettyName = pretty

classifyRefs
  :: Set.Set String            -- primitive/symbol set from YAML
  -> Map.Map String String     -- alias -> "Common::GF28 as GF28"
  -> Set.Set String            -- top-level def names
  -> Set.Set String            -- params (bound vars)
  -> Set.Set String            -- locals (where/let bound defs)
  -> Set.Set String            -- raw referenced strings
  -> [Ref]
classifyRefs primSet aliasMap topDefs paramVars localDefs refs =
  [ classifyOne r
  | r <- Set.toList refs
  ]
  where
    classifyOne r
      | r `Set.member` paramVars = RefVar r
      | r `Set.member` localDefs = RefLocal r
      | r `Set.member` topDefs   = RefDef r
      | r `Set.member` primSet   = RefPrim r
      | otherwise =
          case classifyImport aliasMap r of
            Just x  -> x
            Nothing -> RefOp r

classifyImport :: Map.Map String String -> String -> Maybe Ref
classifyImport aliasMap r =
  case Map.lookup r aliasMap of
    Just imp -> Just (RefImport imp)
    Nothing  ->
      case breakOn "::" r of
        Just (pref, mem) ->
          case Map.lookup pref aliasMap of
            Just imp ->
              if null mem
                then Just (RefImport imp)
                else Just (RefImportMember imp mem)
            Nothing -> Nothing
        Nothing -> Nothing


--------------------------------------------------------------------------------
-- MCC graph builder
--------------------------------------------------------------------------------

data GBState = GBState
  { gbNext  :: Int
  , gbNodes :: [MCCNode]
  , gbEdges :: [MCCEdge]
  }

type Build a = State GBState a

freshNode :: String -> String -> [String] -> Build Int
freshNode k lbl det = do
  nid <- getsNext
  modify' $ \st ->
    st { gbNext  = gbNext st + 1
       , gbNodes = MCCNode nid k lbl det : gbNodes st
       }
  pure nid
  where
    getsNext = do
      st <- Control.Monad.State.Strict.get
      pure (gbNext st)

addEdge :: Int -> Int -> String -> Maybe String -> Build ()
addEdge a b k lab =
  modify' $ \st ->
    st { gbEdges = MCCEdge a b k lab : gbEdges st }

buildMCCForExpr
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Set.Set String
  -> Set.Set String
  -> String
  -> Expr PName
  -> MCCGraph
buildMCCForExpr primSet aliasMap topDefs paramVars localDefs defNameStr expr =
  let st0 = GBState 0 [] []
      (g, stF) = runState (do
        entry <- freshNode "entry" defNameStr []
        (eIn, eOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs expr
        exit  <- freshNode "exit" "exit" []
        addEdge entry eIn "control" Nothing
        addEdge eOut exit "control" Nothing
        pure (entry, exit)
        ) st0
      (entryId, exitId) = g
  in MCCGraph
       { mccEntry = entryId
       , mccExit  = exitId
       , mccNodes = reverse (gbNodes stF)
       , mccEdges = reverse (gbEdges stF)
       }

buildMCCForGuards
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Set.Set String
  -> Set.Set String
  -> String
  -> [PropGuardCase PName]
  -> MCCGraph
buildMCCForGuards primSet aliasMap topDefs paramVars localDefs defNameStr cases =
  let st0 = GBState 0 [] []
      ((entryId, exitId), stF) =
        runState
          (do
             entry <- freshNode "entry" defNameStr []
             d     <- freshNode "guards" "prop_guards" ["cases=" ++ show (length cases)]
             join  <- freshNode "join" "join" []
             exit  <- freshNode "exit" "exit" []

             addEdge entry d "control" Nothing

             forM_ (zip [0 :: Int ..] cases) $ \(i, c) -> do
               let propsStr =
                     intercalate ", " [ pretty (thing p) | p <- pgcProps c ]
                   lab =
                     if null propsStr
                       then Just ("case " ++ show i)
                       else Just ("case " ++ show i ++ ": " ++ propsStr)

               (cin, cout) <- buildExpr primSet aliasMap topDefs paramVars localDefs (pgcExpr c)
               addEdge d cin "branch" lab
               addEdge cout join "control" Nothing

             addEdge d join "branch" (Just "no_match")
             addEdge join exit "control" Nothing
             pure (entry, exit)
          )
          st0
  in MCCGraph
       { mccEntry = entryId
       , mccExit  = exitId
       , mccNodes = reverse (gbNodes stF)
       , mccEdges = reverse (gbEdges stF)
       }

buildExpr
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Set.Set String
  -> Set.Set String
  -> Expr PName
  -> Build (Int, Int)
buildExpr primSet aliasMap topDefs paramVars localDefs expr =
  case expr of
    EVar n -> do
      let s = pretty n
          k = classifyNameKind primSet aliasMap topDefs paramVars localDefs s
      nid <- freshNode k s []
      pure (nid, nid)

    ELit _ -> do
      nid <- freshNode "literal" "lit" []
      pure (nid, nid)

    EIf c t f -> do
      (cIn, cOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs c
      d <- freshNode "if" "if" []
      addEdge cOut d "control" Nothing

      (tIn, tOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs t
      (fIn, fOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs f

      j <- freshNode "join" "join" []
      addEdge d tIn "branch" (Just "then")
      addEdge d fIn "branch" (Just "else")
      addEdge tOut j "control" Nothing
      addEdge fOut j "control" Nothing
      pure (cIn, j)

    ECase scrut alts -> do
      (sIn, sOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs scrut
      d <- freshNode "case" "case" ["alts=" ++ show (length alts)]
      addEdge sOut d "control" Nothing
      j <- freshNode "join" "join" []

      sequence_
        [ do
            let plab = pretty pat
            (aIn, aOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs rhs
            addEdge d aIn "branch" (Just ("alt: " ++ plab))
            addEdge aOut j "control" Nothing
        | CaseAlt pat rhs <- alts
        ]

      pure (sIn, j)

    EComp body mss -> do
      -- Model comprehension as a single loop decision to support McCabe (adds a back-edge).
      let matches = concat mss
          boundFromMatches = Set.unions (map matchBinds matches)
          -- locals/vars introduced by matches become "vars" in the body
          paramVars' = paramVars `Set.union` boundFromMatches

      -- Build generator source expressions as a linear prelude (metadata-ish but kept as control)
      (preIn, preOut) <- buildMatches primSet aliasMap topDefs paramVars localDefs matches

      loop <- freshNode "comprehension" "comp" ["arms=" ++ show (length mss), "matches=" ++ show (length matches)]
      addEdge preOut loop "control" Nothing

      (bIn, bOut) <- buildExpr primSet aliasMap topDefs paramVars' localDefs body
      j <- freshNode "join" "join" []

      addEdge loop bIn "branch" (Just "iter")
      addEdge bOut loop "back"  (Just "next")
      addEdge loop j "branch"   (Just "done")

      pure (preIn, j)

    EWhere e ds -> do
      let localDecls = Set.unions (map declBinds ds)
          localDeclList = sort (Set.toList localDecls)
          localDefs' = localDefs `Set.union` localDecls

      w <- freshNode "where" "where" localDeclList

      -- Add metadata edges to declared local names (non-control)
      sequence_
        [ do
            dn <- freshNode "declaration" nm []
            addEdge w dn "decl" (Just "declares")
        | nm <- localDeclList
        ]

      (eIn, eOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs' e
      addEdge w eIn "control" Nothing
      pure (w, eOut)

    EInfix a op _ b -> do
      (aIn, aOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs a
      (bIn, bOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs b
      let opName = pretty (thing op)
          opKind = if opName `Set.member` primSet then "prim_op" else "operator"
      on <- freshNode opKind opName []
      addEdge aOut bIn "control" Nothing
      addEdge bOut on "control" Nothing
      pure (aIn, on)

    EApp f x -> chain2 "apply" "apply" f x
    EAppT e _ -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    ETyped e _ -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    ELocated e _ -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    EParens e    -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    ESplit e     -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    EPrefix _ e  -> buildExpr primSet aliasMap topDefs paramVars localDefs e

    ETuple es    -> chainMany "tuple" "tuple" es
    EList es     -> chainMany "list" "list" es
    ERecord rec  -> chainMany "record" "record" [ e | (_,e) <- recordElements rec ]
    ESel e _     -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    EUpd me fields ->
      let base = maybe [] (:[]) me
          fs   = [ e | UpdField _ _ e <- fields ]
      in chainMany "update" "update" (base ++ fs)

    EInfFrom e me ->
      case me of
        Nothing -> buildExpr primSet aliasMap topDefs paramVars localDefs e
        Just e2 -> chain2 "range" "inf_from" e e2

    -- Ranges: treat as atomic for MCC purposes
    EFromTo{}         -> atom "range" "from_to"
    EFromToBy{}       -> atom "range" "from_to_by"
    EFromToDownBy{}   -> atom "range" "from_to_downby"
    EFromToLessThan{} -> atom "range" "from_to_lt"
    EGenerate e       -> buildExpr primSet aliasMap topDefs paramVars localDefs e
    ETypeVal _        -> atom "typeval" "typeval"
    EFun _ _ e        -> buildExpr primSet aliasMap topDefs paramVars localDefs e
  where
    atom k lbl = do
      n <- freshNode k lbl []
      pure (n,n)

    chain2 k lbl a b = do
      (aIn, aOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs a
      (bIn, bOut) <- buildExpr primSet aliasMap topDefs paramVars localDefs b
      n <- freshNode k lbl []
      addEdge aOut bIn "control" Nothing
      addEdge bOut n  "control" Nothing
      pure (aIn, n)

    chainMany k lbl es =
      case es of
        [] -> atom k (lbl ++ ":empty")
        [x] -> buildExpr primSet aliasMap topDefs paramVars localDefs x
        (x:xs) -> do
          (i0,o0) <- buildExpr primSet aliasMap topDefs paramVars localDefs x
          (iN,oN) <- foldlMChain o0 xs
          n <- freshNode k lbl ["arity=" ++ show (length es)]
          addEdge oN n "control" Nothing
          pure (i0, n)

    foldlMChain prevOut [] = pure (prevOut, prevOut)
    foldlMChain prevOut (y:ys) = do
      (yin, yout) <- buildExpr primSet aliasMap topDefs paramVars localDefs y
      addEdge prevOut yin "control" Nothing
      foldlMChain yout ys

buildMatches
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Set.Set String
  -> Set.Set String
  -> [Match PName]
  -> Build (Int, Int)
buildMatches primSet aliasMap topDefs paramVars localDefs ms =
  case ms of
    [] -> do
      n <- freshNode "comp_prelude" "prelude" ["empty"]
      pure (n,n)
    _  -> do
      let buildOne m =
            case m of
              Match pat e -> do
                (ein,eout) <- buildExpr primSet aliasMap topDefs paramVars localDefs e
                pn <- freshNode "match" (pretty pat) []
                addEdge eout pn "control" Nothing
                pure (ein, pn)
              MatchLet b -> do
                let nm = pretty (thing (bName b))
                dn <- freshNode "declaration" nm ["matchlet"]
                pure (dn, dn)

      (i0,o0) <- buildOne (head ms)
      foldlM (\(_,prevOut) m -> do
                (iin,iout) <- buildOne m
                addEdge prevOut iin "control" Nothing
                pure (i0,iout)
             ) (i0,o0) (tail ms)

matchBinds :: Match PName -> Set.Set String
matchBinds m =
  case m of
    Match pat _ -> patBinds pat
    MatchLet b  -> Set.singleton (pretty (thing (bName b)))

declBinds :: Decl PName -> Set.Set String
declBinds d =
  case d of
    DBind b     -> Set.singleton (pretty (thing (bName b)))
    DRec bs     -> Set.fromList [ pretty (thing (bName b)) | b <- bs ]
    DPatBind p _ ->
      case patVarName p of
        Just nm -> Set.singleton (pretty nm)
        Nothing -> Set.empty

    DType ts ->
      Set.singleton (pretty (thing (tsName ts)))

    DLocated d' _ -> declBinds d'
    _            -> Set.empty

localDeclNamesInExpr :: Expr PName -> Set.Set String
localDeclNamesInExpr e =
  case e of
    EWhere e' ds ->
      Set.unions (map declBinds ds) `Set.union`
      localDeclNamesInExpr e' `Set.union`
      Set.unions (map localDeclNamesInDecl ds)
    EComp body mss ->
      let ms = concat mss
      in Set.unions (map matchBinds ms) `Set.union`
         localDeclNamesInExpr body `Set.union`
         Set.unions (map localDeclNamesInMatch ms)
    EIf a b c -> localDeclNamesInExpr a `Set.union` localDeclNamesInExpr b `Set.union` localDeclNamesInExpr c
    ECase x alts ->
      localDeclNamesInExpr x `Set.union` Set.unions [ localDeclNamesInExpr rhs | CaseAlt _ rhs <- alts ]
    EApp a b -> localDeclNamesInExpr a `Set.union` localDeclNamesInExpr b
    EInfix a _ _ b -> localDeclNamesInExpr a `Set.union` localDeclNamesInExpr b
    ETyped x _ -> localDeclNamesInExpr x
    ELocated x _ -> localDeclNamesInExpr x
    EParens x -> localDeclNamesInExpr x
    ESplit x -> localDeclNamesInExpr x
    EPrefix _ x -> localDeclNamesInExpr x
    EAppT x _ -> localDeclNamesInExpr x
    ETuple xs -> Set.unions (map localDeclNamesInExpr xs)
    EList xs  -> Set.unions (map localDeclNamesInExpr xs)
    ERecord rec -> Set.unions [ localDeclNamesInExpr ex | (_,ex) <- recordElements rec ]
    EUpd me fields ->
      maybe Set.empty localDeclNamesInExpr me `Set.union`
      Set.unions [ localDeclNamesInExpr ex | UpdField _ _ ex <- fields ]
    ESel x _ -> localDeclNamesInExpr x
    EGenerate x -> localDeclNamesInExpr x
    EInfFrom x mx -> localDeclNamesInExpr x `Set.union` maybe Set.empty localDeclNamesInExpr mx
    EFun _ _ x -> localDeclNamesInExpr x
    _ -> Set.empty

localDeclNamesInDecl :: Decl PName -> Set.Set String
localDeclNamesInDecl d =
  case d of
    DBind b ->
      case bindImpl b of
        Just (DExpr e)        -> localDeclNamesInExpr e
        Just (DPropGuards cs) -> Set.unions [ localDeclNamesInExpr (pgcExpr c) | c <- cs ]
        _                     -> Set.empty
    DRec bs ->
      Set.unions [ localDeclNamesInDecl (DBind b) | b <- bs ]
    DPatBind _ e ->
      localDeclNamesInExpr e
    DLocated d' _ ->
      localDeclNamesInDecl d'
    _ ->
      Set.empty

localDeclNamesInMatch :: Match PName -> Set.Set String
localDeclNamesInMatch m =
  case m of
    Match _ e -> localDeclNamesInExpr e
    MatchLet b ->
      case bindImpl b of
        Just (DExpr e)        -> localDeclNamesInExpr e
        Just (DPropGuards cs) -> Set.unions [ localDeclNamesInExpr (pgcExpr c) | c <- cs ]
        _                     -> Set.empty

classifyNameKind
  :: Set.Set String
  -> Map.Map String String
  -> Set.Set String
  -> Set.Set String
  -> Set.Set String
  -> String
  -> String
classifyNameKind primSet aliasMap topDefs paramVars localDefs s
  | s `Set.member` paramVars = "var"
  | s `Set.member` localDefs = "local"
  | s `Set.member` topDefs   = "call"
  | s `Set.member` primSet   = "prim"
  | otherwise =
      case classifyImport aliasMap s of
        Just _  -> "import"
        Nothing -> "op"

--------------------------------------------------------------------------------
-- Referenced names inside schemas/types/exprs (yours, unchanged)
--------------------------------------------------------------------------------

namesInSchema :: Schema PName -> Set.Set String
namesInSchema (Forall _ props ty _) =
  Set.map prettyName (namesInType ty `Set.union` Set.unions (map namesInProp props))

namesInProp :: Prop PName -> Set.Set PName
namesInProp (CType ty) = namesInType ty

namesInType :: Type PName -> Set.Set PName
namesInType ty =
  case ty of
    TFun a b       -> namesInType a `Set.union` namesInType b
    TSeq a b       -> namesInType a `Set.union` namesInType b
    TBit           -> Set.empty
    TNum _         -> Set.empty
    TChar _        -> Set.empty
    TUser n args   -> Set.insert (thing n) (Set.unions (map namesInType args))
    TTyApp ts      -> Set.unions [ namesInType (value t) | t <- ts ]
    TRecord fs      -> Set.unions [ namesInType t | (_, t) <- recordElements fs ]
    TTuple ts      -> Set.unions (map namesInType ts)
    TWild          -> Set.empty
    TLocated t _   -> namesInType t
    TParens t _    -> namesInType t
    TInfix a _ _ b -> namesInType a `Set.union` namesInType b

namesInExpr :: Expr PName -> Set.Set PName
namesInExpr expr =
  case expr of
    EVar n        -> Set.singleton n
    ELit _        -> Set.empty
    EGenerate e   -> go e
    ETuple es     -> Set.unions (map go es)
    ERecord rec   -> Set.unions [ go e | (_, e) <- recordElements rec ]
    ESel e _      -> go e
    EUpd me fields ->
      maybe Set.empty go me `Set.union`
      Set.unions [ go e | UpdField _ _ e <- fields ]
    EList es      -> Set.unions (map go es)
    EFromTo{}         -> Set.empty
    EFromToBy{}       -> Set.empty
    EFromToDownBy{}   -> Set.empty
    EFromToLessThan{} -> Set.empty
    EInfFrom e me ->
      go e `Set.union` maybe Set.empty go me
    EComp e mss ->
      go e `Set.union` Set.unions [ Set.unions (map namesInMatch ms) | ms <- mss ]
    EApp e1 e2   -> go e1 `Set.union` go e2
    EAppT e _    -> go e
    EIf e1 e2 e3 -> go e1 `Set.union` go e2 `Set.union` go e3
    ECase e alts ->
      go e `Set.union` Set.unions [ go rhs | CaseAlt _ rhs <- alts ]
    EWhere e ds  ->
      go e `Set.union` Set.unions (map namesInDecl ds)
    ETyped e _   -> go e
    ETypeVal _   -> Set.empty
    EFun _ _ e   -> go e
    ELocated e _ -> go e
    ESplit e     -> go e
    EParens e    -> go e
    EInfix e1 op _ e2 ->
      go e1 `Set.union` go e2 `Set.union` Set.singleton (thing op)
    EPrefix _ e  -> go e
  where
    go = namesInExpr

namesInMatch :: Match PName -> Set.Set PName
namesInMatch m =
  case m of
    Match _ e  -> namesInExpr e
    MatchLet b -> namesInDecl (DBind b)

namesInDecl :: Decl PName -> Set.Set PName
namesInDecl d =
  case d of
    DBind b ->
      case bindImpl b of
        Just (DExpr e)        -> namesInExpr e
        Just (DPropGuards cs) -> Set.unions [ namesInExpr (pgcExpr c) | c <- cs ]
        Nothing               -> Set.empty
    DRec bs ->
      Set.unions [ namesInDecl (DBind b) | b <- bs ]
    DPatBind _ e ->
      namesInExpr e
    DSignature _ _ ->
      Set.empty
    DType (TySyn _ _ _ ty) ->
      namesInType ty
    DLocated d' _ ->
      namesInDecl d'
    _ ->
      Set.empty

dropLocDecl :: Decl PName -> Decl PName
dropLocDecl (DLocated d _) = dropLocDecl d
dropLocDecl d              = d

--------------------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------------------

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

breakOn :: String -> String -> Maybe (String, String)
breakOn pat s =
  case findIndexSubstring pat s of
    Nothing -> Nothing
    Just i  ->
      let (l, r0) = splitAt i s
          r       = drop (length pat) r0   -- drop the matched pattern
      in Just (l, r)
  where
    findIndexSubstring :: String -> String -> Maybe Int
    findIndexSubstring p xs = go 0 xs
      where
        go _ [] = Nothing
        go i ys
          | p `isPrefixOf` ys = Just i
          | otherwise         = go (i+1) (tail ys)
